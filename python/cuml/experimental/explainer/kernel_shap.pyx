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

import cuml
import cuml.internals
import cupy as cp
import numpy as np

from cuml.common.import_utils import has_shap
from cuml.common.import_utils import has_sklearn
from cuml.common.input_utils import input_to_cuml_array
from cuml.common.input_utils import input_to_cupy_array
from cuml.common.logger import info
from cuml.common.logger import warn
from cuml.experimental.explainer.base import SHAPBase
from cuml.experimental.explainer.common import get_cai_ptr
from cuml.experimental.explainer.common import model_func_call
from cuml.linear_model import Lasso
from cuml.raft.common.handle import Handle
from functools import lru_cache
from itertools import combinations
from random import randint

from cuml.raft.common.handle cimport handle_t
from libc.stdint cimport uintptr_t
from libc.stdint cimport uint64_t


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
        uint64_t seed) except +

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
        uint64_t seed) except +


class KernelExplainer(SHAPBase):
    """
    GPU accelerated of SHAP's kernel explainer, optimized for tabular data.
    Based on the SHAP package:
    https://github.com/slundberg/shap/blob/master/shap/explainers/_kernel.py

    Main differences of the GPU version:

    - Data generation and Kernel SHAP calculations are significantly faster,
    but this has a tradeoff of having more model evaluations if both the
    observation explained and the background data have many 0-valued columns.
    - There is a small initialization cost (similar to training time of regular
    Scikit/cuML models), which was a tradeoff for faster explanations after
    that.
    - Only tabular data is supported for now, via passing the background
    dataset explicitly. Since the new API of SHAP is still evolving, the main
    supported API right now is the old one
    (i.e. ``explainer.shap_values()``)
    - Sparse data support is planned for the near future.
    - Further optimizations are in progress.

    Parameters
    ----------
    model : function
        Function that takes a matrix of samples (n_samples, n_features) and
        computes the output for those samples with shape (n_samples). Function
        must use either CuPy or NumPy arrays as input/output.
    data : Dense matrix containing floats or doubles.
        cuML's kernel SHAP supports tabular data for now, so it expects
        a background dataset, as opposed to a shap.masker object.
        The background dataset to use for integrating out features.
        To determine the impact of a feature, that feature is set to "missing"
        and the change in the model output is observed.
        Acceptable formats: CUDA array interface compliant objects like
        CuPy, cuDF DataFrame/Series, NumPy ndarray and Pandas
        DataFrame/Series.
    nsamples : int (default = 2 * data.shape[1] + 2048)
        Number of times to re-evaluate the model when explaining each
        prediction. More samples lead to lower variance estimates of the SHAP
        values. The "auto" setting uses `nsamples = 2 * X.shape[1] + 2048`.
    link : function or str (default = 'identity')
        The link function used to map between the output units of the
        model and the SHAP value units.
    random_state: int, RandomState instance or None (default = None)
        Seed for the random number generator for dataset creation.
    gpu_model : bool or None (default = None)
        If None Explainer will try to infer whether `model` can take GPU data
        (as CuPy arrays), otherwise it will use NumPy arrays to call `model`.
        Set to True to force the explainer to use GPU data,  set to False to
        force the Explainer to use NumPy data.
    handle : cuml.raft.common.handle
        Specifies the handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    dtype : np.float32 or np.float64 (default = None)
        Parameter to specify the precision of data to generate to call the
        model. If not specified, the explainer will try to get the dtype
        of the model, if it cannot be queried, then it will defaul to
        np.float32.
    output_type : 'cupy' or 'numpy' (default = None)
        Parameter to specify the type of data to output.
        If not specified, the explainer will try to see if model is gpu based,
        if so it will be set to `cupy`, otherwise it will be set to `numpy`.
        For compatibility with SHAP's graphing libraries, specify `numpy`.

    Examples
    --------

    >>> from cuml import SVR
    >>> from cuml import make_regression
    >>> from cuml import train_test_split
    >>>
    >>> from cuml.experimental.explainer import KernelExplainer as cuKE
    >>>
    >>> X, y = make_regression(
    ...     n_samples=102,
    ...     n_features=10,
    ...     noise=0.1,
    ...     random_state=42)

    >>>
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X,
    ...     y,
    ...     test_size=2,
    ...     random_state=42)
    >>>
    >>> model = SVR().fit(X_train, y_train)
    >>>
    >>> cu_explainer = cuKE(
    ...     model=model.predict,
    ...     data=X_train,
    ...     gpu_model=True)
    >>>
    >>> cu_shap_values = cu_explainer.shap_values(X_test)
    >>> cu_shap_values
    array([[ 0.02104662, -0.03674018, -0.01316485,  0.02408933, -0.5943235 ,
             0.15274985, -0.01287319, -0.3050412 ,  0.0262317 , -0.07229283],
           [ 0.15244992,  0.16341315, -0.09833339,  0.07259235, -0.17099564,
             2.7372282 ,  0.0998467 , -0.29607034, -0.11780564, -0.50097287]],
          dtype=float32)

    """

    @cuml.internals.api_return_any()
    def __init__(self,
                 *,
                 model,
                 data,
                 nsamples=None,
                 link='identity',
                 verbose=False,
                 random_state=None,
                 gpu_model=None,
                 handle=None,
                 dtype=None,
                 output_type=None):

        super(KernelExplainer, self).__init__(
            model=model,
            data=data,
            order='C',
            link=link,
            verbose=verbose,
            random_state=random_state,
            gpu_model=True,
            handle=handle,
            dtype=dtype,
            output_type=output_type
        )

        # Matching SHAP package default values for number of samples
        self.nsamples = 2 * self.M + 2 ** 11 if nsamples is None else nsamples

        # Maximum number of samples that user can set
        max_samples = 2 ** 32

        # restricting maximum number of samples
        if self.M <= 32:
            max_samples = 2 ** self.M - 2

            # if the user requested more samples than there are subsets in the
            # _powerset, we set nsamples to max_samples
            if self.nsamples > max_samples:
                info("`nsamples` exceeds maximum number of samples {}, "
                     "setting it to that value.".format(max_samples))
                self.nsamples = max_samples

        # Check the ratio between samples we evaluate divided by
        # all possible samples to check for need for l1
        self.ratio_evaluated = self.nsamples / max_samples

        self.nsamples_exact, self.nsamples_random, self.randind = \
            self._get_number_of_exact_random_samples(data=data,
                                                     ncols=self.M,
                                                     nsamples=self.nsamples)

        # using numpy for powerset and shapley kernel weight calculations
        # cost is incurred only once, and generally we only generate
        # very few samples of the powerset if M is big.
        mat, weight = _powerset(self.M, self.randind - 1, self.nsamples_exact,
                                dtype=self.dtype)

        # Store the mask and weights as device arrays
        # Mask dtype can be independent of Explainer dtype, since model
        # is not called on it.
        self._mask = cp.zeros((self.nsamples, self.M), dtype=np.float32)
        self._mask[:self.nsamples_exact] = cp.array(mat)

        self._weights = cp.ones(self.nsamples, dtype=self.dtype)
        self._weights[:self.nsamples_exact] = cp.array(weight)

        self._synth_data = None

        # evaluate the model in background to get the expected_value
        self.expected_value = self.link_fn(
            cp.mean(
                model_func_call(X=self.background,
                                model_func=self.model,
                                model_gpu_based=self.model_gpu_based)
            )
        )

    def _get_number_of_exact_random_samples(self, data, ncols, nsamples):
        """
        Function calculates how many rows will be from the powerset (exact)
        and how many will be from random samples, based on the nsamples
        of the explainer.
        """
        cur_nsamples = 0
        nsamples_exact = 0
        r = 0

        # we check how many subsets of the _powerset of self.M we can fit
        # in self.nsamples. This sets of the powerset are used  as indexes
        # to generate the mask matrix
        while cur_nsamples <= self.nsamples:
            r += 1
            nsamples_exact = cur_nsamples
            cur_nsamples += int(_binomCoef(self.M, r))

        # see if we need to have randomly sampled entries in our mask
        # and combinations matrices
        nsamples_random = \
            nsamples - nsamples_exact if r < ncols else 0

        # we save r so we can generate random samples later
        randind = r

        return nsamples_exact, nsamples_random, r

    def shap_values(self,
                    X,
                    l1_reg='auto'):
        """
        Interface to estimate the SHAP values for a set of samples.
        Corresponds to the SHAP package's legacy interface, and is our main
        API currently.

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
        return self._explain(X, l1_reg)

    def __call__(self,
                 X,
                 l1_reg='auto'):
        """
        Experimental interface to estimate the SHAP values for a set of
        samples.
        Corresponds to the SHAP package's new API, building a SHAP.Explanation
        object for the result. It is experimental, it is recommended to use
        `Explainer.shap_values` during the first version.

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
        if has_shap("0.36"):
            warn("SHAP's Explanation object is still experimental, the main "
                 "API currently is `explainer.shap_values`.")
            from shap import Explanation
            res = self._explain(X, l1_reg)
            out = Explanation(
                values=res,
                base_values=self.expected_value,
                data=self.background,
                feature_names=self.feature_names,
            )
            return out
        else:
            raise ImportError("SHAP >= 0.36 package required to build "
                              "Explanation object. Use the "
                              "`explainer.shap_values` function to get "
                              "the shap values, or install "
                              "SHAP to use the new API style.")

    @cuml.internals.api_return_array()
    def _explain(self,
                 X,
                 nsamples=None,
                 l1_reg='auto'):
        if X.ndim == 1:
            X = X.reshape((1, self.M))

        shap_values = cp.zeros(X.shape, dtype=self.dtype)

        # Allocate synthetic dataset array once for multiple explanations
        if self._synth_data is None:
            self._synth_data = cp.zeros(
                shape=(self.N * self.nsamples, self.M),
                dtype=self.dtype,
                order=self.order
            )

        # Explain each observation
        idx = 0
        for x in X:
            shap_values[idx] = self._explain_single_observation(
                x.reshape(1, self.M), l1_reg
            )
            idx = idx + 1

        return shap_values[0]

    def _explain_single_observation(self,
                                    row,
                                    l1_reg):
        # Call the model to get the value f(row)
        self.fx = cp.array(
            model_func_call(X=row,
                            model_func=self.model,
                            model_gpu_based=self.model_gpu_based))

        # If we need sampled rows, then we call the function that generates
        # the samples array with how many samples each row will have
        # and its corresponding weight
        if self.nsamples_random > 0:
            samples, self._weights[self.nsamples_exact:self.nsamples] = \
                self._generate_nsamples_weights()

        row, n_rows, n_cols, dtype = \
            input_to_cuml_array(row, order=self.order)

        cdef handle_t* handle_ = \
            <handle_t*><size_t>self.handle.getHandle()
        cdef uintptr_t row_ptr, bg_ptr, cmb_ptr, masked_ptr, x_ptr, smp_ptr

        row_ptr = row.ptr
        bg_ptr = get_cai_ptr(self.background)
        cmb_ptr = get_cai_ptr(self._synth_data)
        if self.nsamples_random > 0:
            smp_ptr = get_cai_ptr(samples)
        else:
            smp_ptr = <uintptr_t> NULL
            maxsample = 0

        x_ptr = get_cai_ptr(self._mask)

        if self.random_state is None:
            random_state = randint(0, 1e18)

        # we default to float32 unless self.dtype is specifically np.float64
        if self.dtype == np.float64:
            kernel_dataset(
                handle_[0],
                <double*> x_ptr,
                <int> self._mask.shape[0],
                <int> self._mask.shape[1],
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
                <int> self._mask.shape[0],
                <int> self._mask.shape[1],
                <float*> bg_ptr,
                <int> self.background.shape[0],
                <float*> cmb_ptr,
                <float*> row_ptr,
                <int*> smp_ptr,
                <int> self.nsamples_random,
                <int> maxsample,
                <uint64_t> random_state)

        # evaluate model on combinations
        y = model_func_call(X=self._synth_data,
                            model_func=self.model,
                            model_gpu_based=self.model_gpu_based)

        # get average of each combination of X
        y_hat = cp.mean(
            cp.array(y).reshape((self.nsamples,
                                 self.background.shape[0])),
            axis=1
        )

        nonzero_inds = self._l1_regularization(y_hat, l1_reg)

        return self._weighted_linear_regression(y_hat, nonzero_inds)

    def _generate_nsamples_weights(self):
        """
        Function generates an array `samples` of ints of samples and their
        weights that can be used for generating X and dataset.
        """
        samples = np.random.choice(np.arange(self.randind,
                                             self.randind + 1),
                                   self.nsamples_random)
        maxsample = np.max(samples)
        w = np.empty(self.nsamples_random, dtype=self.dtype)
        for i in range(self.nsamples_exact, self.nsamples_random):
            w[i] = shapley_kernel(self.M, samples[i])
        samples = cp.array(samples, dtype=np.int32)
        w = cp.array(w)
        return samples, w

    def _l1_regularization(self, y_hat, l1_reg):
        """
        Function calls LASSO or LARS if l1 regularization is needed.
        """
        nonzero_inds = None
        # call lasso/lars if needed
        if l1_reg == 'auto':
            if self.ratio_evaluated < 0.2:
                # todo: analyze ideal alpha if staying with lasso or switch
                # to cuml lars once that is merged
                nonzero_inds = cp.nonzero(
                    Lasso(
                        alpha=0.1,
                        handle=self.handle,
                        verbosity=self.verbosity).fit(
                            X=self._mask,
                            y=y_hat
                    ).coef_)[0]
                if len(nonzero_inds) == 0:
                    return cp.zeros(self.M)

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
                        self._mask, y_hat, max_iter=r)[1]
                elif (isinstance(l1_reg, str) and l1_reg == "bic" or
                        l1_reg == "aic"):
                    nonzero_inds = np.nonzero(
                        LassoLarsIC(criterion=l1_reg).fit(self._mask,
                                                          y_hat).coef_)[0]
        return nonzero_inds

    def _weighted_linear_regression(self, y_hat, nonzero_inds=None):
        """
        Function performs weighted linear regression, the shap values
        are the coefficients.
        """
        if nonzero_inds is None:
            y_hat = y_hat - self.expected_value
            Aw = self._mask * cp.sqrt(self._weights[:, cp.newaxis])
            Bw = y_hat * cp.sqrt(self._weights)

        else:
            y_hat = y_hat[nonzero_inds] - self.expected_value

            Aw = self._mask[nonzero_inds] * cp.sqrt(
                self._weights[nonzero_inds, cp.newaxis]
            )

            Bw = y_hat * cp.sqrt(self._weights[nonzero_inds])

        X, *_ = cp.linalg.lstsq(Aw, Bw)
        return X


@lru_cache(maxsize=None)
def _binomCoef(n, k):
    """
    Binomial coefficient function with cache
    """
    res = 1
    if(k > n - k):
        k = n - k
    for i in range(k):
        res *= (n - i)
        res /= (i + 1)

    return res


def _powerset(n, r, nrows, dtype=np.float32):
    """
    Function to generate the subsets of range(n) up to size r.
    """
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


def _calc_sampling_weights(M, r):
    """
    Function to calculate sampling weights to
    """
    w = np.empty(M - r, dtype=np.float32)
    for i in range(M - r, M):
        w[i] = (M - 1) / i * (M - i)
    return w


@lru_cache(maxsize=None)
def shapley_kernel(M, s):
    """
    Function that calculates shapley kernel, cached.
    """
    # To avoid infinite values
    # Based on reference implementation
    if(s == 0 or s == M):
        return 10000

    res = (M - 1) / (_binomCoef(M, s) * s * (M - s))
    return res
