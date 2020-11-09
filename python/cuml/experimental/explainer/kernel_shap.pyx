#
# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import cupy as cp
import numpy as np

from cudf import DataFrame as cu_df
from cuml.common.array import CumlArray
from cuml.common.import_utils import has_scipy
from cuml.common.import_utils import has_sklearn
from cuml.common.input_utils import input_to_cuml_array
from cuml.experimental.explainer.common import get_model_order_from_tags
from cuml.experimental.explainer.common import link_dict
from cuml.linear_model import Lasso
from functools import cache
from pandas import DataFrame as pd_df
from itertools import combinations


cdef extern from "cuml/explainer/kernel_shap.hpp" namespace "ML":
    kernel_dataset "ML::Explainer::kernel_dataset"(
        handle_t& handle,
        int* X,
        int nrows_X,
        int M,
        float* background,
        int nrows_background,
        float* combinations,
        float* observation,
        int* nsamples,
        int len_nsamples,
        int maxsample)

    kernel_dataset "ML::Explainer::kernel_dataset"(
        handle_t& handle,
        int* X,
        int nrows_X,
        int M,
        double* background,
        int nrows_background,
        double* combinations,
        double* observation,
        int* nsamples,
        int len_nsamples,
        int maxsample)


class KernelSHAP():
    """

    GPU accelerated of SHAP's kernel explainer:
    https://github.com/slundberg/shap/blob/master/shap/explainers/_kernel.py

    Main differences of the GPU version:

    - Data generation and Kernel SHAP calculations are significantly faster,
    but this has a tradeoff of having more model evaluations if the observation
    explained has the same entries as background observations.
    - There is an initialization cost (similar to training time of regular
    Scikit/cuML models), which was a tradeoff for faster explanations after
    that.
    - Only tabular data is supported for now, via passing the background
    dataset explicitly. Since the new API of SHAP is still evolving, the main
    supported API right now is the old one
    (i.e. explainer.shap_values())
    - Sparse data support is in progress.
    - Further optimizations are in progress.

    Parameters
    ----------
    model : function
        A callable python object that executes the model given a set of input
        data samples.
    data : Dense matrix containing floats or doubles.
        cuML's kernel SHAP supports tabular data for now, so it expects
        a background dataset, as opposed to a shap.masker object. To respect
        a hierarchical structure of the data, use the (temporary) parameter
        'masker_type'
        Acceptable formats: CUDA array interface compliant objects like
        CuPy, cuDF DataFrame/Series, NumPy ndarray and Pandas
        DataFrame/Series.
    nsamples : int
        Number of samples to use to estimate shap values.
    masker_type: {'independent', 'partition'} default = 'independent'
        If 'independent' is used, then this is equivalent to SHAP's
        independent masker and the algorithm is fully GPU accelerated.
        If 'partition' then it is equivalent to SHAP's Partition masker,
        which respects a hierarchical structure in the background data.
    link : function or str
        The link function used to map between the output units of the
        model and the SHAP value units.

    """

    def __init__(self,
                 model,
                 data,
                 nsamples=None,
                 link='identity',
                 verbosity=False):
        # validate and save the link function
        if isinstance(link, str):
            # todo: add check if link is link_dict
            self.link = link_dict[link]
        elif callable(link) and callable(getattr(link, "inverse", None)):
            self.link = link
        else:
            raise TypeError("`link` function is not valid.")

        self.model = model

        self.order = get_model_order_from_tags(model=model, default='C')

        self.background = input_to_cuml_array(data, order=self.order)
        self.M = data.shape[1]
        self.N = data.shape[0]

        self.nsamples = 2 * self.M + 2 ** 1 if nsamples is None else nsamples

        self.max_samples = 2 ** 30

        # restricting maximum number of samples for memory and performance
        # value being checked
        if self.M <= 30:
            self.max_samples = 2 ** self.M - 2
            if self.nsamples > self.max_samples:
                self.nsamples = self.max_samples

        if isinstance(data, pd_df) or isinstance(data, cu_df):
            self.feature_names = data.columns.to_list()
        else:
            self.feature_names = [None for _ in range(len(data))]

        cur_nsamples = self.M
        r = 1
        while cur_nsamples < nsamples:
            if has_scipy():
                from scipy.special import binom
                cur_nsamples += int(binom(self.M, r))
            else:
                cur_nsamples += int(binomCoef(self.M, r))
            r += 1

        # using numpy powerset and calculations for initial version
        # cost is incurred once
        mat = powerset(self.M, r, nsamples)

        self.X = CumlArray(mat)
        self.nexact = len(self.exact_mask)
        self.nsampled = max(nsamples - cur_nsamples, 0)
        self.X.append(cp.zeros((self.n_sampled, self.M)))

        self.weights = cp.ones(len(self.X))
        self._combinations = None

        self.expected_value = self.link(cp.sum(model(self.background)))

    def explain(self,
                X,
                nsamples=None,
                l1_reg):
        shap_values = cp.zeros((len(X), self.n_cols), dtype=self.dtype)

        if self._combinations is None:
            # self._y_hat = CumlArray.zeros(
            #     shape=(self.nsamples, 1),
            #     dtype=np.float32
            # )
            self._combinations = CumlArray.zeros(
                shape=(self.N * self.nsamples, self.M),
                dtype=np.float32
            )

        idx = 0
        for x in X:
            shap_values[idx] = self._explain_single_observation(x, l1_reg)
            idx += 1

    def _explain_single_observation(self,
                                    row,
                                    l1_reg):

        # np choice of weights - for samples if needed
        if self.nsampled > 0:
            samples = np.random.choice(len(self.weights),
                                       4 * nsampled, p=self.weights)
            maxsample = np.max(samples)
            samples = CumlArray(samples)

        row = row.reshape(1, self.n_cols)
        row, n_rows, n_cols, dtype = \
            input_to_cuml_array(row, order=self.order)

        cdef handle_t* handle_ = \
            <handle_t*><size_t>self.handle.getHandle()
        cdef uintptr_t row_ptr, bg_ptr, cmb_ptr, masked_ptr

        row_ptr = row.ptr
        bg_ptr = self.background.ptr
        cmb_ptr = self._combinations.ptr
        smp_ptr = samples.ptr

        # todo: add dtype check

        kernel_dataset(
            handle_[0],
            <int*> self.X,
            <int> X.shape[0],
            <int> X.shape[1],
            <float*> bg_ptr,
            <int> self.background.shape[0],
            <float*> cmb_ptr,
            <float*> row_ptr,
            <int*> nsamps_ptr,
            <int> self.nsampled,
            <int> maxsample)

        # evaluate model on combinations

        y = self.model(self._combinations)
        y_hat = cp.mean(cp.array(y).reshape((self.nsamples,
                                             self.background.shape[0])))

        averaged_outs = cp.mean(cp.asarray(self.link(self._y)), axis=1)

        # call lasso/lars if needed
        if(l1_reg == 'auto' and self.nsamples / self.max_samples < 0.2):
            nonzero_inds = cp.nonzero(
                Lasso(alpha=l1_reg).fit(self.X, y_hat).coef_
            )[0]
            if len(nonzero_inds) == 0:
                return cp.zeros(self.M), np.ones(self.M)

        else:
            if not has_sklearn():
                raise ImportError("Scikit-learn needed for lars l1 "
                                  "regularization currently.")
            else:
                # todo: raise warning

                from sklearn.linear_model import LassoLarsIC, lars_path
                if (isinstance(l1_reg, str)
                        and self.l1_reg.startswith("num_features(")):
                    r = int(self.l1_reg[len("num_features("):-1])
                    nonzero_inds = lars_path(
                        self.X, y_hat, max_iter=r)[1]
                elif (isinstance(l1_reg, str) and l1_reg == "bic" or
                        self.l1_reg == "aic"):
                    nonzero_inds = np.nonzero(LassoLarsIC(criterion=c).fit(mask_aug, eyAdj_aug).coef_)[0]

        # weighted linear regression
        tmp = cp.linalg.inv(cp.dot(cp.dot(self.X.T, np.diag(self.weights)),
                                   self.X))

        return cp.dot(tmp, cp.dot(cp.dot(self.X.T, cp.diag(self.weights)),
                                  self._y))

    def shap_values(self, X, l1_reg='auto'):
        """
        Legacy interface to estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : Dense matrix containing floats or doubles.
            Acceptable formats: CUDA array interface compliant objects like
            CuPy, cuDF DataFrame/Series, NumPy ndarray and Pandas
            DataFrame/Series.
        l1_reg : str (default: 'auto')
            The l1 regularization to use for feature selection.

        Returns
        -------
        array or list

        """
        return self.explain(X, l1_reg)

    def __call__(self, X, l1_reg='auto'):
        # todo: add explanation object construction
        return self.explain(X, l1_reg)


@cache
def binomCoef(n, k):
    res = 1
    if(k > n - k):
        k = n - k
    for i in range(k):
        res *= (n - i)
        res /= (i + 1)

    return res


def powerset(n, r, nrows):
    N = np.arange(n)
    result = np.zeros((nrows, n))
    idx = 0
    for i in range(1, r + 1):
        for c in combinations(N, i):
            print(c)
            result[idx, c] = 1
            idx += 1

    return result


@cache
shapley_kernel(M, s):
    if(s == 0 or s == M):
        return 10000

    res = (M - 1) / (binomCoef(M, s) * s * (M - s))
    return res
