from typing import List

import numpy as np
import itertools
import sklearn.neighbors
import functools
import joblib
import psutil
import warnings
import pandas as pd
from tqdm import tqdm

from ..base import BaseExplainer
from ._cadex_parallel import compute_criterion  # type: ignore


class ECE(BaseExplainer):
    def __init__(self, k: int, columns: List[str], bces: List[BaseExplainer], dist: int, h: int,
                 lambda_: float, n_jobs=None):
        self._col_names = columns
        self.k = k
        self.bces = bces
        self.norm = dist
        self.h = h
        self.lambda_ = np.float32(lambda_)
        if n_jobs is None:
            self.n_jobs = psutil.cpu_count(logical=False)
        else:
            self.n_jobs = n_jobs
        self._cfs_len: int
        #self._aggregated_cfs: NDArray[np.float32]
        self._aggregated_cfs: np.ndarray

    #def _aggregate_cfs(self, x) -> NDArray[np.float32]:
    def _aggregate_cfs(self, x) -> np.ndarray:
        #list_cfs: List[NDArray[np.float32]] = []
        list_cfs: List[np.ndarray] = []
        list_cfs_explainers: List[str] = []
        for bce in self.bces:
            #bce_result: NDArray[np.float32] = np.asarray(bce.generate(x).values)
            res = bce.generate(x)
            if isinstance(res, pd.DataFrame):
                bce_result: np.ndarray = np.asarray(res.values)
                for bce_r in bce_result:
                    list_cfs.append(bce_r)
                    list_cfs_explainers.append(str(bce))
                print(f'BCE {bce} generated {len(bce_result)} counterfactuals')
            else:
                print(f'BCE {bce} found no countefactuals')

        #cfs: NDArray[np.float32] = np.unique(np.asarray(list_cfs), axis=0)
        cfs, indexes = np.unique(np.asarray(list_cfs), axis=0, return_index=True)
        list_cfs_explainers = np.array(list_cfs_explainers)[indexes]
        self._cfs_len = cfs.shape[0]
        assert isinstance(cfs, np.ndarray)
        return cfs, list_cfs_explainers

    #def _choose_best_k(self, valid_cfs: NDArray[np.float32], x_series):
    def _choose_best_k(self, valid_cfs: np.ndarray, x_series):
        x = x_series.values
        #orms: NDArray[np.float32] = np.apply_along_axis(functools.partial(np.linalg.norm, ord=self.norm), 0, valid_cfs)
        norms: np.ndarray = np.apply_along_axis(functools.partial(np.linalg.norm, ord=self.norm), 0, valid_cfs)
        norms = np.nan_to_num(norms)
        #C = list(valid_cfs / norms)
        # C = np.where(norms==0, 0, valid_cfs/norms)
        with np.errstate(divide='ignore', invalid='ignore'):
            C = np.true_divide(valid_cfs, norms)
            C[C == np.inf] = 0
            C = np.nan_to_num(C)
        # C = np.nan_to_num(C)
        k = min(self.k, self._cfs_len)
        if k != self.k:
            warnings.warn(f'k parameter > number of aggregated counterfactuals. Changing k from {self.k} to {k}',
                          UserWarning, stacklevel=3)
        if self._cfs_len <= self.h:
            warnings.warn(
                f"knn's h parameter >= number of aggregated counterfactuals. Changing h from  {self.h} to {self._cfs_len - 1}",
                UserWarning, stacklevel=3)
            self.h = self._cfs_len - 1
        k_subsets = list()
        for i in range(k):
            k_subsets += list(itertools.combinations(C, r=i + 1))

        # Take only some subsets as we cannot evaluate too much of them
        np.random.shuffle(k_subsets)
        k_subsets = k_subsets[:min(len(k_subsets), 100)]

        knn_c = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
        #c_np: NDArray[np.float32] = np.asarray(C)
        c_np: np.ndarray = np.asarray(C)
        knn_c.fit(c_np, np.ones(shape=c_np.shape[0]))

        # S_ids = joblib.Parallel(n_jobs=self.n_jobs)(
        #     joblib.delayed(compute_criterion)(knn_c, self.norm, self.lambda_, c_np, x, S) for S in k_subsets)
        # selected = norms * k_subsets[np.argmax(np.asarray(S_ids))]

        S_ids = []
        for i in tqdm(range(len(k_subsets)), desc='Computing criterion for k subset selection'):
            S_ids += [compute_criterion(knn_c, self.norm, self.lambda_, c_np, x, k_subsets[i])]
        selected = norms * k_subsets[np.argmax(np.asarray(S_ids))]
        return selected

    def generate(self, x: pd.Series) -> pd.DataFrame:
        self._aggregated_cfs, list_cfs_explainers = self._aggregate_cfs(x)
        #print(self._aggregated_cfs)
        # Remove cfs with NaNs
        self._aggregated_cfs  = self._aggregated_cfs[~np.isnan(self._aggregated_cfs).any(axis=1)]
        list_cfs_explainers = list_cfs_explainers[~np.isnan(self._aggregated_cfs).any(axis=1)]
        #print(self._aggregated_cfs)
        # k_subset = self._choose_best_k(self._aggregated_cfs, x)

        # # Preserve mapping of conterfactuals and their explainers
        # return_list_cfs_explainer_mapping = list()
        # for row in k_subset:
        #     for idx, row_original in enumerate(self._aggregated_cfs):
        #         if np.all(np.equal(row, row_original)):
        #             return_list_cfs_explainer_mapping.append(list_cfs_explainers[idx])

        # return pd.DataFrame(k_subset, columns=self._col_names), return_list_cfs_explainer_mapping
        return pd.DataFrame(self._aggregated_cfs, columns=self._col_names), list_cfs_explainers

    def get_aggregated_len(self):
        if self._cfs_len is None:
            raise AttributeError('Aggregation has not been performed yet')
        return self._cfs_len

    def get_aggregated_cfs(self):
        if self._aggregated_cfs is None:
            raise AttributeError('Aggregation has not been performed yet')
        return pd.DataFrame(self._aggregated_cfs, columns=self._col_names)
