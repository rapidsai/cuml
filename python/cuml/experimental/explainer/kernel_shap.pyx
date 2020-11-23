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
from cuml.common.import_utils import has_shap
from cuml.common.import_utils import has_sklearn
from cuml.common.input_utils import input_to_cuml_array
from cuml.common.input_utils import input_to_cupy_array
from cuml.common.logger import info
from cuml.common.logger import warn
from cuml.experimental.explainer.common import get_dtype_from_model_func
from cuml.experimental.explainer.common import get_link_fn_from_str
from cuml.experimental.explainer.common import get_tag_from_model_func
from cuml.experimental.explainer.common import link_dict
from cuml.experimental.explainer.common import model_call
from cuml.linear_model import Lasso
from cuml.raft.common.handle import Handle
from functools import lru_cache
from pandas import DataFrame as pd_df
from itertools import combinations
from random import randint

from cuml.raft.common.handle cimport handle_t
from libc.stdint cimport uintptr_t
from libc.stdint cimport uint64_t

from pdb import set_trace


cdef extern from "cuml/explainer/kernel_shap.hpp" namespace "ML":
    void kernel_dataset "ML::Explainer::kernel_dataset"(
        handle_t& handle,
        float* X,
        int nrows_X,
        int ncols,
        float* background,
        int nrows_background,
        float* combinations,
        float* observation,
        int* nsamples,
        int len_nsamples,
        int maxsample,
        uint64_t seed)

    void kernel_dataset "ML::Explainer::kernel_dataset"(
        handle_t& handle,
        double* X,
        int nrows_X,
        int ncols,
        double* background,
        int nrows_background,
        double* combinations,
        double* observation,
        int* nsamples,
        int len_nsamples,
        int maxsample,
        uint64_t seed)


class KernelExplainer():
    """
    GPU accelerated of SHAP's kernel explainer:
    https://github.com/slundberg/shap/blob/master/shap/explainers/_kernel.py

    Main differences of the GPU version:

    - Data generation and Kernel SHAP calculations are significantly faster,
    but this has a tradeoff of having more model evaluations if both the
    observation explained and the background data have many 0-valued columns.
    - There is an initialization cost (similar to training time of regular
    Scikit/cuML models), which was a tradeoff for faster explanations after
    that.
    - Only tabular data is supported for now, via passing the background
    dataset explicitly. Since the new API of SHAP is still evolving, the main
    supported API right now is the old one
    (i.e. explainer.shap_values())
    - Sparse data support is not yet implemented.
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
    random_state: int, RandomState instance or None (default)
        Seed for the random number generator for dataset creation.
    gpu_model : bool

    handle : cuml.raft.common.handle
        Specifies the handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    dtype : np.float32 or np.float64 (default=None)
        Parameter to specify the precision of data to generate to call the
        model. If not specified, the explainer will try to get the dtype
        of the model, if it cannot be queried, then it will defaul to
        np.float32.
    output_type : 'cupy' or 'numpy' (default:None)
        Parameter to specify the type of data to output.
        If not specified, the explainer will try to see if model is gpu based,
        if so it will default to `cupy`, otherwise it will default to `numpy`.
        For compatibility with SHAP's graphing libraries, specify `numpy`.

    """

    def __init__(self,
                 model,
                 data,
                 nsamples=None,
                 link='identity',
                 verbosity=False,
                 random_state=None,
                 gpu_model=None,
                 handle=None,
                 dtype=None,
                 output_type=None):

        self.handle = Handle() if handle is None else handle

        self.link = link
        self.link_fn = get_link_fn_from_str(link)
        self.model = model
        self.order = get_tag_from_model_func(func=model,
                                             tag='preferred_input_order',
                                             default='C')
        if gpu_model is None:
            # todo: when sparse support is added, use this tag to see if
            # model can accept sparse data
            self.model_gpu_based = \
                get_tag_from_model_func(func=model,
                                        tag='X_types_gpu',
                                        default=False) is not None
        else:
            self.model_gpu_based = gpu_model

        if output_type is None:
            self.output_type = 'cupy' if self.model_gpu_based else 'numpy'
        else:
            self.output_type = output_type

        # if not dtype is specified, we try to get it from the model
        if dtype is None:
            self.dtype = get_dtype_from_model_func(func=model,
                                                   default=np.float32)
        else:
            self.dtype = np.dtype(dtype)

        self.background, self.N, self.M, _ = \
            input_to_cupy_array(data, order='C',
                                convert_to_dtype=self.dtype)

        self.nsamples = 2 * self.M + 2 ** 12 if nsamples is None else nsamples

        self.max_samples = 2 ** 30

        # restricting maximum number of samples for memory and performance
        # value being checked, right now based on mainline SHAP package
        self.max_samples = 2 ** 30
        if self.M <= 30:
            self.max_samples = 2 ** self.M - 2
            if self.nsamples > self.max_samples:
                self.nsamples = self.max_samples

        if isinstance(data, pd_df) or isinstance(data, cu_df):
            self.feature_names = data.columns.to_list()
        else:
            self.feature_names = [None for _ in range(len(data))]

        cur_nsamples = 0
        self.nsamples_exact = 0
        r = 0

        # we check how many subsets of the powerset of self.M we can fit
        # in self.nsamples
        while cur_nsamples <= self.nsamples:
            r += 1
            self.nsamples_exact = cur_nsamples
            cur_nsamples += int(binomCoef(self.M, r))

        # see if we need to have randomly sampled entries in our mask
        # and combinations matrices
        self.nsamples_random = \
            self.nsamples - self.nsamples_exact if r < self.M else 0

        # we save r so we can generate random samples later
        self.randind = r

        # using numpy powerset and calculations for initial version
        # cost is incurred only once, and generally we only generate
        # very few samples if M is big.
        mat, weight = powerset(self.M, r - 1, self.nsamples_exact,
                               dtype=self.dtype)

        self.mask = cp.zeros((self.nsamples, self.M), dtype=np.float32)
        self.mask[:self.nsamples_exact] = cp.array(mat)

        self.weights = cp.ones(self.nsamples, dtype=self.dtype)
        self.weights[:self.nsamples_exact] = cp.array(weight)

        self.synth_data = None

        self.expected_value = self.link_fn(cp.mean(model(self.background)))

        self.random_state = random_state

    def explain(self,
                X,
                nsamples=None,
                l1_reg='auto'):
        shap_values = cp.zeros((1, self.M), dtype=self.dtype)

        if X.ndim == 1:
            X = X.reshape((1, self.M))

        # allocating combinations array once for multiple explanations
        if self.synth_data is None:
            self.synth_data = cp.zeros(
                shape=(self.N * self.nsamples, self.M),
                dtype=np.float32,
                order='C'
            )

        idx = 0
        for x in X:
            shap_values[idx] = self._explain_single_observation(
                x.reshape(1, self.M), l1_reg
            )

        if isinstance(X, np.ndarray):
            out_type = 'numpy'
        else:
            out_type = 'cupy'
        return input_to_cuml_array(shap_values)[0].to_output(out_type)

    def _explain_single_observation(self,
                                    row,
                                    l1_reg):
        self.fx = cp.array(
            model_call(X=row,
                       model=self.model,
                       model_gpu_based=self.model_gpu_based))

        if self.nsamples_random > 0:
            samples, self.weights[self.nsamples_exact:self.nsamples] = \
                self._generate_number_samples_weights()

        row, n_rows, n_cols, dtype = \
            input_to_cuml_array(row, order=self.order)

        cdef handle_t* handle_ = \
            <handle_t*><size_t>self.handle.getHandle()
        cdef uintptr_t row_ptr, bg_ptr, cmb_ptr, masked_ptr, x_ptr, smp_ptr

        row_ptr = row.ptr
        bg_ptr = self.background.__cuda_array_interface__['data'][0]
        cmb_ptr = self.synth_data.__cuda_array_interface__['data'][0]
        if self.nsamples_random > 0:
            smp_ptr = samples.__cuda_array_interface__['data'][0]
        else:
            smp_ptr = <uintptr_t> NULL
            maxsample = 0

        x_ptr = self.mask.__cuda_array_interface__['data'][0]

        if self.random_state is None:
            random_state = randint(0, 1e18)

        # we default to float32 unless self.dtype is specifically np.float64
        if self.dtype == np.float64:
            kernel_dataset(
                handle_[0],
                <double*> x_ptr,
                <int> self.mask.shape[0],
                <int> self.mask.shape[1],
                <double*> bg_ptr,
                <int> self.background.shape[0],
                <double*> cmb_ptr,
                <double*> row_ptr,
                <int*> smp_ptr,
                <int> self.nsamples_random,
                <int> maxsample,
                <uint64_t> random_state)

        else:

            kernel_dataset(
                handle_[0],
                <float*> x_ptr,
                <int> self.mask.shape[0],
                <int> self.mask.shape[1],
                <float*> bg_ptr,
                <int> self.background.shape[0],
                <float*> cmb_ptr,
                <float*> row_ptr,
                <int*> smp_ptr,
                <int> self.nsamples_random,
                <int> maxsample,
                <uint64_t> random_state)

        # evaluate model on combinations
        self.y = model_call(X=self.synth_data,
                            model=self.model,
                            model_gpu_based=self.model_gpu_based)

        # get average of each combination of X
        y_hat = cp.mean(
            cp.array(self.y).reshape((self.nsamples,
                                      self.background.shape[0])),
            axis=1
        )

        nonzero_inds = None

        # call lasso/lars if needed
        if l1_reg == 'auto':
            if self.nsamples / self.max_samples < 0.2:
                # todo: analyze ideal alpha if staying with lasso or switch
                # to cuml lars once that is merged
                nonzero_inds = cp.nonzero(
                    Lasso(alpha=0.25).fit(self.mask, y_hat).coef_
                )[0]
                if len(nonzero_inds) == 0:
                    return cp.zeros(self.M), np.ones(self.M)

        else:
            if not has_sklearn():
                raise ImportError("Scikit-learn needed for lars l1 "
                                  "regularization currently.")
            else:
                warn("LARS is not currently GPU accelerated, using "
                     "Scikit-learn.")

                from sklearn.linear_model import LassoLarsIC, lars_path
                if (isinstance(l1_reg, str)
                        and l1_reg.startswith("num_features(")):
                    r = int(l1_reg[len("num_features("):-1])
                    nonzero_inds = lars_path(
                        self.mask, y_hat, max_iter=r)[1]
                elif (isinstance(l1_reg, str) and l1_reg == "bic" or
                        l1_reg == "aic"):
                    nonzero_inds = np.nonzero(
                        LassoLarsIC(criterion=l1_reg).fit(self.mask,
                                                          y_hat).coef_)[0]

        return self._weighted_linear_regression(y_hat, nonzero_inds)

    def _generate_number_samples_weights(self):
        """
        Function generates an array `samples` of ints of samples and their
        weights that can be used for generating X and dataset.
        """
        samples = np.random.choice(np.arange(self.randind,
                                             self.randind + 3),
                                   self.nsamples_random)
        maxsample = np.max(samples)
        w = np.empty(self.nsamples_random, dtype=np.float32)
        for i in range(self.nsamples_exact, self.nsamples_random):
            w[i] = shapley_kernel(self.M, samples[i])
        samples = cp.array(samples, dtype=np.int32)
        w = cp.array(w)
        return samples, w

    def _weighted_linear_regression(self, y_hat, nonzero_inds=None):
        if nonzero_inds is None:
            y_hat = y_hat - self.expected_value

            Aw = self.mask * cp.sqrt(self.weights[:, cp.newaxis])
            Bw = y_hat * cp.sqrt(self.weights)
            X, *_ = cp.linalg.lstsq(Aw, Bw)

            return X
        else:
            y_hat = y_hat[nonzero_inds] - self.expected_value
            Aw = self.mask[nonzero_inds] * cp.sqrt(
                self.weights[nonzero_inds, cp.newaxis]
            )
            Bw = y_hat * cp.sqrt(self.weights[nonzero_inds])
            X, *_ = cp.linalg.lstsq(Aw, Bw)

            return X

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

    def __call__(self,
                 X,
                 l1_reg='auto'):
        if has_shap():
            warn("SHAP's Explanation object is still experimental, the main "
                 "API currently is 'explainer.shap_values'.")
            from shap import Explanation
            res = self.explain(X, l1_reg)
            out = Explanation(
                values=res,
                base_values=self.expected_value,
                base_values=self.expected_value,
                data=self.background,
                feature_names=self.feature_names,
            )
            return out
        else:
            raise ImportError("SHAP package required to build Explanation "
                              "object. Use the explainer.shap_values "
                              "function to get the shap values, or install "
                              "SHAP to use the new API style.")


@lru_cache(maxsize=None)
def binomCoef(n, k):
    res = 1
    if(k > n - k):
        k = n - k
    for i in range(k):
        res *= (n - i)
        res /= (i + 1)

    return res


def powerset(n, r, nrows, dtype=np.float32):
    N = np.arange(n)
    w = np.zeros(nrows, dtype=dtype)
    result = np.zeros((nrows, n), dtype=dtype)
    idx = 0
    for i in range(1, r + 1):
        for c in combinations(N, i):
            result[idx, c] = 1
            w[idx] = shapley_kernel(n, i)
            idx += 1

    return result, w


def calc_sample_weights(M, r):
    w = np.empty(M - r, dtype=np.float32)
    for i in range(M - r, M):
        # w[i] = shapley_kernel(nsamples, i)
        w[i] = (M - 1) / i * (M - i)
    return w


@lru_cache(maxsize=None)
def shapley_kernel(M, s):
    if(s == 0 or s == M):
        return 10000

    res = (M - 1) / (binomCoef(M, s) * s * (M - s))
    return res
