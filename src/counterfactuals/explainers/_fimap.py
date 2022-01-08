import operator
from typing import Tuple, List, Any, Optional

import numpy as np
import pandas as pd
import sklearn.model_selection
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.layers import ReLU
from numpy.typing import NDArray
from tensorflow.keras.layers import Layer, Lambda, ActivityRegularization, Dense, Dropout, Input, Add, Concatenate, \
    Multiply
from sklearn.preprocessing import LabelBinarizer
from counterfactuals.base import BaseExplainer

from counterfactuals.constraints import Freeze, ValueNominal, ValueMonotonicity, OneHot
from counterfactuals.preprocessing import DataFrameMapper


def _freeze_layers(model: tf.keras.Model) -> None:
    for layer in model.layers:
        layer.trainable = False


def _build_s(input_shape) -> tf.keras.Model:
    x = Input(shape=input_shape)
    y = Dense(200, activation='relu')(x)
    y = Dropout(0.2)(y)
    y = Dense(200, activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Dense(200, activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Dense(1, activation='sigmoid')(y)

    s = tf.keras.Model(inputs=x, outputs=y)

    return s


def _freeze_constraints_to_mask(layer_size: int, freeze_constraints: List[Freeze], dtype='float32') \
        -> NDArray[np.float32]:
    mask = np.zeros(shape=(layer_size,), dtype=dtype)
    for constraint in freeze_constraints:
        columns = np.asarray(constraint.columns)
        mask[columns] = 1
    return mask


def _gumbel_distribution(shape):
    u_dist = K.random_uniform(tf.shape(shape), 0, 1)
    return -K.log(-K.log(u_dist + K.epsilon()) + K.epsilon())


class _GumbelSoftmax(Layer):

    def __init__(self, tau: float, num_classes: int):
        super(_GumbelSoftmax, self).__init__()
        self.tau = tau
        self.num_classes = num_classes

    def call(self, inputs, **kwargs):
        x = inputs + _gumbel_distribution(inputs)
        x = K.softmax(x / self.tau)
        return K.stop_gradient(K.one_hot(K.argmax(x), self.num_classes))


def _get_span(inputs: tf.Tensor, start: int, end: int) -> tf.Tensor:
    return Lambda(lambda x: x[:, start:end])(inputs)


def _get_freeze_mask(shape, constraints: List[Any], mapper: DataFrameMapper, dtype="float32") -> NDArray[np.float32]:
    mask = np.ones(shape, dtype=dtype)
    for constraint in constraints:
        if isinstance(constraint, Freeze):
            for column in constraint.columns:
                start, end = mapper.transformed_column_span(column)
                mask[start:end] = 0.
    return mask


def _get_freeze_columns(constraints: List[Any], mapper: DataFrameMapper) -> List[int]:
    columns = []
    for constraint in constraints:
        if isinstance(constraint, Freeze):
            for column in constraint.columns:
                columns.extend(mapper.transformed_column_span(column))
    return columns


class MyL1Regularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, l1: float):
        self.l1 = l1

    def __call__(self, x):
        original = Lambda(lambda _x: _x[:, :, 0])(x)
        x = Lambda(lambda _x: _x[:, :, 1])(x)
        not_equals = K.stop_gradient(
            K.cast(
                K.not_equal(
                    K.argmax(original, axis=1), K.argmax(x, axis=1)),
                dtype=tf.float32
            )
        )
        return self.l1 * tf.reduce_sum(tf.abs(x) * not_equals)


def _build_g(input_shape,
             layers: List[Any],
             one_hot_columns: List[Tuple[int, int]],
             increasing_columns: List[int],
             decreasing_columns: List[int],
             freezed_columns: List[int],
             l1: float, l2: float, tau: float
             ):
    input_shape = np.prod(input_shape)
    inputs = Input((input_shape,))
    dense1 = Dense(100, activation='relu')(inputs)
    dropout1 = Dropout(0.2)(dense1)
    dense2 = Dense(100, activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(dense2)
    dense3 = Dense(input_shape)(dropout2)

    continuous_columns = {
        i: Lambda(lambda x: x[:, i:i+1])(dense3)
        for i in range(input_shape) if not any(i in range(start, end) for (start, end) in one_hot_columns)
    }

    for i, col in continuous_columns.items():
        if i in increasing_columns:
            continuous_columns[i] = ReLU()(col)
        elif i in decreasing_columns:
            continuous_columns[i] = ReLU(max_value=0., negative_slope=-1.)(col)
        elif i in freezed_columns:
            continuous_columns[i] = ReLU(max_value=0., negative_slope=0.)(col)

    for i, col in continuous_columns.items():
        continuous_columns[i] = Add()([
            ActivityRegularization(l2=l2)(col),
            Lambda(lambda _x: _x[:, i:i + 1])(inputs)
        ])

    print(continuous_columns)

    one_hot_columns_layers = {
        (start, end): Lambda(lambda x: x[:, start:end]) for (start, end) in one_hot_columns
    }

    print(one_hot_columns_layers)

    for (start, end), cols in one_hot_columns_layers.items():
        if any([i in range(start, end) for i in freezed_columns]):
            one_hot_columns_layers[(start, end)] = cols(inputs)
        else:
            cols = _GumbelSoftmax(tau=tau, num_classes=end - start)(cols(dense3))
            one_hot_columns_layers[(start, end)] = cols
            regularizer_branch = tf.stack([Lambda(lambda _x: _x[:, start:end])(inputs), cols], axis=0)
            print(tf.shape(regularizer_branch))
            MyL1Regularizer(l1=l1)(regularizer_branch)

    if one_hot_columns_layers and continuous_columns:
        concat1 = Concatenate(axis=-1)(list(continuous_columns.values()))
        concat2 = Concatenate(axis=-1)(list(one_hot_columns_layers.values()))
        output = Concatenate(axis=-1)([concat1, concat2])
    elif continuous_columns:
        output = Concatenate(axis=-1)(list(continuous_columns.values()))
    else:
        output = Concatenate(axis=-1)(list(one_hot_columns_layers.values()))

    g = tf.keras.Model(inputs=inputs, outputs=output)

    return g


def _fit_g_s(s, g, x, y, epochs):
    if g is not None:
        print("\nTraining g")
    else:
        print("\nTraining s")
    optimizer = tf.keras.optimizers.Adam(0.001)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    acc = tf.keras.metrics.BinaryAccuracy()
    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    batch_size = x_train.shape[0]
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size)

    for epoch in range(epochs):
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                if g is not None:
                    x_perturbed = g(x_batch_train, training=True)
                    s_pred = s(x_perturbed, training=True)
                else:
                    s_pred = s(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, s_pred)
                if g is not None:
                    loss_value += tf.reduce_sum(g.losses)  # l1 and l2 regularization
                acc.update_state(y_batch_train, s_pred)

            if g is not None:
                g_grads = tape.gradient(loss_value, g.trainable_weights)
                optimizer.apply_gradients(zip(g_grads, g.trainable_weights))
            else:
                s_grads = tape.gradient(loss_value, s.trainable_weights)
                optimizer.apply_gradients(zip(s_grads, s.trainable_weights))
        if (epoch + 1) % 10 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value)),
                "\nTraining accuracy", acc.result().numpy()
            )


def _get_nominal_columns_span(constraints: List[Any], mapper: DataFrameMapper) -> List[Tuple[int, int]]:
    columns = []
    for constraint in constraints:
        if isinstance(constraint, ValueNominal):
            for column in constraint.columns:
                columns.append(mapper.transformed_column_span(column))
    return columns


def _get_nominal_columns(constraints: List[Any]) -> List[str]:
    columns = []
    for constraint in constraints:
        if isinstance(constraint, ValueNominal):
            columns.extend(constraint.columns)
    return columns


def _get_one_hot_columns(constraints: List[Any]) -> List[Tuple[int, int]]:
    columns = []
    for constraint in constraints:
        if isinstance(constraint, OneHot):
            columns.append((constraint.start_column, constraint.end_column))
    return columns


def _get_continuous_columns(columns: List[str], nominal_columns: List[str]) -> List[str]:
    return [column for column in columns if column not in nominal_columns]


def _get_increasing_columns(constraints: List[Any], mapper: DataFrameMapper) -> List[int]:
    columns = []
    for constraint in constraints:
        if isinstance(constraint, ValueMonotonicity) and constraint.direction == 'increasing':
            columns.extend(constraint.columns)

    return [mapper.transformed_column_span(col)[0] for col in columns]


def _get_decreasing_columns(constraints: List[Any], mapper: DataFrameMapper) -> List[int]:
    columns = []
    for constraint in constraints:
        if isinstance(constraint, ValueMonotonicity) and constraint.direction == 'decreasing':
            columns.extend(constraint.columns)

    return [mapper.transformed_column_span(col)[0] for col in columns]


class Fimap(BaseExplainer):

    def __init__(self, tau: float = 0.1, l1: float = 0.01, l2: float = 0.1, constraints: Optional[List[Any]] = None,
                 s: Optional[tf.keras.Model] = None, g_layers: Optional[List[tf.keras.layers.Layer]] = None):
        self._constraints = constraints if constraints is not None else []
        self._s = s
        self._g: tf.keras.Model = None
        self._g_layers = g_layers
        self._sg: tf.keras.Model = None
        self._input_shape: Tuple[int]
        self._assert_not_nominal_and_one_hot()
        self._one_hot_columns = _get_one_hot_columns(self._constraints)
        self._nominal_columns = _get_nominal_columns(self._constraints)
        self._mapper = DataFrameMapper(nominal_columns=self._nominal_columns, one_hot_columns=self._one_hot_columns)
        self._nominal_columns_spans = _get_nominal_columns_span(self._constraints, self._mapper)
        self._continuous_columns: List[str]
        self._tau = tau
        self._l1 = l1
        self._l2 = l2
        self._y_label_binarizer = LabelBinarizer()
        self._increasing_columns = _get_increasing_columns(self._constraints, self._mapper)
        self._decreasing_columns = _get_decreasing_columns(self._constraints, self._mapper)

    def fit(self, x: pd.DataFrame, y: pd.Series, epochs: int = 5, **kwargs) -> None:
        x = self._mapper.fit_transform(x)
        y = self._y_label_binarizer.fit_transform(y)
        input_shape = x.shape[1:]
        s = self._s
        if s is None:
            s = _build_s(input_shape=input_shape)
            _fit_g_s(s, None, x, y, epochs)
        freeze_columns = _get_freeze_columns(self._constraints, self._mapper)
        g = _build_g(input_shape=input_shape,
                     layers=self._g_layers,
                     one_hot_columns=self._mapper.one_hot_spans,
                     freezed_columns=freeze_columns,
                     increasing_columns=self._increasing_columns,
                     decreasing_columns=self._decreasing_columns,
                     l1=self._l1, l2=self._l2,
                     tau=self._tau)
        _fit_g_s(s, g, x, 1 - y, epochs)
        self._s = s
        self._g = g

    def generate(self, x: pd.Series) -> pd.DataFrame:
        x = self._mapper.transform(x.to_frame().T)
        perturbed = self._g.predict(x)
        print(perturbed)
        return self._mapper.inverse_transform(perturbed)

    def _assert_not_nominal_and_one_hot(self):
        if any(isinstance(c, ValueNominal) for c in self._constraints) and any(
                isinstance(c, OneHot) for c in self._constraints):
            raise ValueError(f"You passed {ValueNominal.__name__} and {OneHot.__name__} in constraints, but only one "
                             f"of them should be used")
