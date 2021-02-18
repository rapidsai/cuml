#
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

import cuml.common.logger as logger
import cupy as cp
import numpy as np
import time

from cuml.common.import_utils import has_shap
from cuml.common.import_utils import has_sklearn
from cuml.common.input_utils import input_to_cupy_array
from cuml.experimental.explainer.base import SHAPBase
from cuml.experimental.explainer.common import get_cai_ptr
from cuml.experimental.explainer.common import model_func_call
from cuml.experimental.explainer.common import output_list_shap_values
from cuml.linear_model import Lasso
from cuml.linear_model import LinearRegression
from cuml.raft.common.handle import Handle
from functools import lru_cache
from itertools import combinations
from numbers import Number
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
        float* X,
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
    GPU accelerated of SHAP's kernel explainer (experimental).

    Based on the SHAP package:
    https://github.com/slundberg/shap/blob/master/shap/explainers/_kernel.py

    Main differences of the GPU version:

     - Data generation and Kernel SHAP calculations are significantly faster,
       but this has a tradeoff of having more model evaluations if both the
       observation explained and the background data have many 0-valued
       columns.
     - Support for SHAP's new Explanation and API will be available in the
       next version.
     - There is a small initialization cost (similar to training time of
       regular Scikit/cuML models) of a few seconds, which was a tradeoff for
       faster explanations after that.
     - Only tabular data is supported for now, via passing the background
       dataset explicitly. Since the new API of SHAP is still evolving, the
       main supported API right now is the old one
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
        values. The "auto" setting uses ``nsamples = 2 * X.shape[1] + 2048``.
    link : function or str (default = 'identity')
        The link function used to map between the output units of the
        model and the SHAP value units. From the SHAP package: The link
        function used to map between the output units of the model and the
        SHAP value units. By default it is identity, but logit can be useful
        so that expectations are computed in probability units while
        explanations remain in the (more naturally additive) log-odds units.
        For more details on how link functions work see any overview of link
        functions for generalized linear models.
    random_state: int, RandomState instance or None (default = None)
        Seed for the random number generator for dataset creation. Note: due to
        the design of the sampling algorithm the concurrency can affect
        results so currently 100% deterministic execution is not guaranteed.
    gpu_model : bool or None (default = None)
        If None Explainer will try to infer whether `model` can take GPU data
        (as CuPy arrays), otherwise it will use NumPy arrays to call `model`.
        Set to True to force the explainer to use GPU data,  set to False to
        force the Explainer to use NumPy data.
    handle : cuml.raft.common.handle (default = None)
        Specifies the handle that holds internal CUDA state for
        computations in this model, a new one is created if it is None.
        Most importantly, this specifies the CUDA stream that will be used for
        the model's computations, so users can run different models
        concurrently in different streams by creating handles in several
        streams.
    dtype : np.float32 or np.float64 (default = None)
        Parameter to specify the precision of data to generate to call the
        model. If not specified, the explainer will try to get the dtype
        of the model, if it cannot be queried, then it will defaul to
        np.float32.
    output_type : 'cupy' or 'numpy' (default = 'numpy')
        Parameter to specify the type of data to output.
        If not specified, the explainer will default to 'numpy' for the time
        being to improve compatibility.

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

    def __init__(self,
                 *,
                 model,
                 data,
                 nsamples=2**11,
                 link='identity',
                 verbose=False,
                 random_state=None,
                 is_gpu_model=None,
                 handle=None,
                 dtype=None,
                 output_type=None):

        super(KernelExplainer, self).__init__(
            model=model,
            background=data,
            order='C',
            link=link,
            verbose=verbose,
            random_state=random_state,
            is_gpu_model=is_gpu_model,
            handle=handle,
            dtype=dtype,
            output_type=output_type
        )

        self.nsamples = nsamples
        # Maximum number of samples that user can set
        max_samples = 2 ** 32

        # restricting maximum number of samples
        if self.ncols <= 32:
            max_samples = 2 ** self.ncols - 2

            # if the user requested more samples than there are subsets in the
            # _powerset, we set nsamples to max_samples
            if self.nsamples > max_samples:
                self.nsamples = max_samples

        # Check the ratio between samples we evaluate divided by
        # all possible samples to check for need for l1
        self.ratio_evaluated = self.nsamples / max_samples

        self.nsamples_exact, self.nsamples_random, self.randind = \
            _get_number_of_exact_random_samples(ncols=self.ncols,
                                                nsamples=self.nsamples)

        # using numpy for powerset and shapley kernel weight calculations
        # cost is incurred only once, and generally we only generate
        # very few samples of the powerset if M is big.
        mat, weight = _powerset(self.ncols, self.randind, self.nsamples_exact,
                                full_powerset=(self.nsamples_random == 0),
                                dtype=self.dtype)

        # Store the mask and weights as device arrays
        # Mask dtype can be independent of Explainer dtype, since model
        # is not called on it.
        self._mask = cp.zeros((self.nsamples, self.ncols), dtype=np.float32)
        self._mask[:self.nsamples_exact] = cp.array(mat)

        self._weights = cp.ones(self.nsamples, dtype=self.dtype)
        self._weights[:self.nsamples_exact] = cp.array(weight)

        self._reset_timers()

    def shap_values(self,
                    X,
                    l1_reg='auto',
                    as_list=True):
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
        as_list : bool (default = True)
            Set to True to return a list of arrays for multi-dimensional
            models (like predict_proba functions) to match the SHAP package
            behavior. Set to False to return them as an array of arrays.

        Returns
        -------
        values : array or list

        """
        return self._explain(X,
                             synth_data_shape=(self.nrows * self.nsamples,
                                               self.ncols),
                             return_as_list=as_list,
                             l1_reg=l1_reg)

    def _explain_single_observation(self,
                                    shap_values,
                                    row,
                                    idx,
                                    l1_reg):
        total_timer = time.time()
        # Call the model to get the value f(row)
        fx = cp.array(
            model_func_call(X=row,
                            model_func=self.model,
                            gpu_model=self.is_gpu_model))

        self.model_call_time = \
            self.model_call_time + (time.time() - total_timer)

        self._mask[self.nsamples_exact:self.nsamples] = \
            cp.zeros((self.nsamples_random, self.ncols), dtype=cp.float32)

        # If we need sampled rows, then we call the function that generates
        # the samples array with how many samples each row will have
        # and its corresponding weight
        if self.nsamples_random > 0:
            samples, self._weights[self.nsamples_exact:self.nsamples] = \
                _generate_nsamples_weights(self.ncols,
                                           self.nsamples,
                                           self.nsamples_exact,
                                           int(self.nsamples_random / 2),
                                           self.randind,
                                           self.dtype)

        row, n_rows, n_cols, dtype = \
            input_to_cupy_array(row, order=self.order)

        cdef handle_t* handle_ = \
            <handle_t*><size_t>self.handle.getHandle()
        cdef uintptr_t row_ptr, bg_ptr, ds_ptr, masked_ptr, x_ptr, smp_ptr

        row_ptr = get_cai_ptr(row)
        bg_ptr = get_cai_ptr(self.background)
        ds_ptr = get_cai_ptr(self._synth_data)
        if self.nsamples_random > 0:
            smp_ptr = get_cai_ptr(samples)
        else:
            smp_ptr = <uintptr_t> NULL
            maxsample = 0

        x_ptr = get_cai_ptr(self._mask)

        if self.random_state is None:
            self.random_state = randint(0, 1e18)

        # we default to float32 unless self.dtype is specifically np.float64
        if self.dtype == np.float64:
            kernel_dataset(
                handle_[0],
                <float*> x_ptr,
                <int> self._mask.shape[0],
                <int> self._mask.shape[1],
                <double*> bg_ptr,
                <int> self.background.shape[0],
                <double*> ds_ptr,
                <double*> row_ptr,
                <int*> smp_ptr,
                <int> self.nsamples_random,
                <int> maxsample,
                <uint64_t> self.random_state)

        else:
            kernel_dataset(
                handle_[0],
                <float*> x_ptr,
                <int> self._mask.shape[0],
                <int> self._mask.shape[1],
                <float*> bg_ptr,
                <int> self.background.shape[0],
                <float*> ds_ptr,
                <float*> row_ptr,
                <int*> smp_ptr,
                <int> self.nsamples_random,
                <int> maxsample,
                <uint64_t> self.random_state)

        # kept while in experimental namespace. It is not needed for cuml
        # models, but for other GPU models it is
        self.handle.sync()

        model_timer = time.time()
        # evaluate model on combinations
        y = model_func_call(X=self._synth_data,
                            model_func=self.model,
                            gpu_model=self.is_gpu_model)

        self.model_call_time = \
            self.model_call_time + (time.time() - model_timer)

        l1_reg_time = 0

        for i in range(self.model_dimensions):
            if self.model_dimensions == 1:
                y_hat = y - self._expected_value
                exp_val_param = self._expected_value
                fx_param = fx[0]
            else:
                y_hat = y[:, i] - self._expected_value[i]
                fx_param = fx[0][i]
                exp_val_param = self._expected_value[i]

            # get average of each combination of X
            y_hat = cp.mean(
                cp.array(y_hat).reshape((self.nsamples,
                                         self.background.shape[0])),
                axis=1
            )

            # we neeed to do l1 regularization if user left it as auto and we
            # evaluated less than 20% of the space, or if the user set it
            # and we did not evaluate all the space (i.e. nsamples_random == 0)
            nonzero_inds = None
            if ((self.ratio_evaluated < 0.2 and l1_reg == "auto") or
                    (self.ratio_evaluated < 1.0 and l1_reg != "auto")):
                reg_timer = time.time()
                nonzero_inds = _l1_regularization(self._mask,
                                                  y_hat,
                                                  self._weights,
                                                  exp_val_param,
                                                  fx_param,
                                                  self.link_fn,
                                                  l1_reg)
                self.l1_reg_time = \
                    self.l1_reg_time + (time.time() - reg_timer)
                # in case all indexes become zero
                if nonzero_inds.shape == (0, ):
                    return None

            reg_timer = time.time()
            shap_values[i][idx, :-1] = _weighted_linear_regression(
                self._mask,
                y_hat,
                self._weights,
                exp_val_param,
                fx_param,
                nonzero_inds=nonzero_inds,
                handle=self.handle)

            # add back the variable that was removed in the weighted
            # linear regression preprocessing
            if nonzero_inds is None:
                shap_values[i][idx, -1] = \
                    (fx_param - exp_val_param) - cp.sum(
                        shap_values[i][idx, :-1])
            else:
                shap_values[i][idx, nonzero_inds[-1]] = \
                    (fx_param - exp_val_param) - cp.sum(
                        shap_values[i][idx, :-1])

            self.linear_model_time = \
                self.linear_model_time + (time.time() - reg_timer)

        self.total_time = self.total_time + (time.time() - total_timer)

    def _reset_timers(self):
        super()._reset_timers()
        self.l1_reg_time = 0
        self.linear_model_time = 0

    def _get_timers_str(self):
        res_str = super()._get_timers_str()
        res_str += "Time spent in L1 regularization: {}".format(
            self.l1_reg_time)
        res_str += "Time spent in linear model calculation: {}".format(
            self.linear_model_time)
        return res_str


def _get_number_of_exact_random_samples(ncols, nsamples):
    """
    Function calculates how many rows will be from the powerset (exact)
    and how many will be from random samples, based on the nsamples
    of the explainer.
    """
    cur_nsamples = 0
    nsamples_exact = 0
    r = 0

    # we check how many subsets of the _powerset of self.ncols we can fit
    # in self.nsamples. This sets of the powerset are used  as indexes
    # to generate the mask matrix
    while cur_nsamples <= nsamples / 2:
        r += 1
        nsamples_exact = cur_nsamples
        cur_nsamples += int(_binomCoef(ncols, r))

    # if we are going to generate a full powerset (i.e. we reached
    # bincoef bincoef(ncols, r/2)) we return 2**ncols - 2
    if r >= ncols / 2:
        nsamples_exact = 2**ncols - 2
    else:
        nsamples_exact *= 2
    # see if we need to have randomly sampled entries in our mask
    # and combinations matrices
    nsamples_random = \
        nsamples - nsamples_exact if r < ncols / 2 else 0

    # we save r so we can generate random samples later
    return nsamples_exact, nsamples_random, r


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


@lru_cache(maxsize=None)
def _shapley_kernel(M, s):
    """
    Function that calculates shapley kernel, cached.
    """
    # To avoid infinite values
    # Based on reference implementation
    if(s == 0 or s == M):
        return 10000

    res = (M - 1) / (_binomCoef(M, s) * s * (M - s))
    return res


def _powerset(n, r, nrows, full_powerset=False, dtype=np.float32):
    """
    Function to generate the subsets of range(n) up to size r.
    """
    N = np.arange(n)
    w = np.zeros(nrows, dtype=dtype)
    result = np.zeros((nrows, n), dtype=dtype)
    idx = 0
    upper_limit = n if full_powerset else r
    for i in range(1, upper_limit):
        for c in combinations(N, i):
            result[idx, c] = 1
            w[idx] = _shapley_kernel(n, i)
            if not full_powerset:
                result[idx + 1] = 1 - result[idx]
                w[idx + 1] = _shapley_kernel(n, i)
                idx += 1
            idx += 1

    return result, w


def _generate_nsamples_weights(ncols,
                               nsamples,
                               nsamples_exact,
                               nsamples_random,
                               randind,
                               dtype):
    """
    Function generates an array `samples` of ints of samples and their
    weights that can be used for generating X and dataset.
    """
    samples = np.random.choice(np.arange(randind,
                                         randind + 2),
                               nsamples_random)
    w = np.empty(nsamples_random * 2, dtype=dtype)
    for i in range(len(samples)):
        weight = \
            _shapley_kernel(ncols, samples[i])
        w[i * 2] = weight
        w[i * 2 + 1] = weight
    samples = cp.array(samples, dtype=np.int32)
    w = cp.array(w)
    return samples, w


def _l1_regularization(X,
                       y,
                       weights,
                       expected_value,
                       fx,
                       link_fn,
                       l1_reg='auto'):
    """
    Function calls LASSO or LARS if l1 regularization is needed.
    """

    # create augmented dataset for feature selection
    s = cp.sum(X, axis=1)
    w_aug = cp.hstack(
        (weights * (X.shape[1] - s), weights * s))
    w_sqrt_aug = np.sqrt(w_aug)
    y = cp.hstack(
        (y, y - (link_fn(fx) - link_fn(expected_value))))
    y *= w_sqrt_aug
    X = cp.transpose(
        w_sqrt_aug * cp.transpose(cp.vstack((X, X - 1))))

    # Use lasso if Scikit-learn is not present
    if not has_sklearn():
        if l1_reg == 'auto':
            l1_reg = 0.2
        elif not isinstance(l1_reg, Number):
            raise ImportError("Scikit-learn is required for l1 "
                              "regularization that is not Lasso.")
        nonzero_inds = cp.nonzero(Lasso(alpha=l1_reg).fit(X, y).coef_)[0]

    # Else match default behavior of mainline SHAP
    elif l1_reg == 'auto':
        from sklearn.linear_model import LassoLarsIC
        nonzero_inds = np.nonzero(
            LassoLarsIC(criterion="aic").fit(cp.asnumpy(X),
                                             cp.asnumpy(y)).coef_)[0]

    elif isinstance(l1_reg, str):
        if l1_reg.startswith("num_features("):
            from sklearn.linear_model import lars_path
            r = int(l1_reg[len("num_features("):-1])
            nonzero_inds = lars_path(cp.asnumpy(X),
                                     cp.asnumpy(y), max_iter=r)[1]
        elif l1_reg in ["aic", "bic"]:
            from sklearn.linear_model import LassoLarsIC
            nonzero_inds = np.nonzero(
                LassoLarsIC(criterion=l1_reg).fit(cp.asnumpy(X),
                                                  cp.asnumpy(y)).coef_)[0]

    else:
        nonzero_inds = cp.nonzero(Lasso(alpha=0.2).fit(X, y).coef_)[0]

    return cp.asarray(nonzero_inds)


def _weighted_linear_regression(X,
                                y,
                                weights,
                                expected_value,
                                fx,
                                nonzero_inds=None,
                                handle=None):
    """
    Function performs weighted linear regression, the shap values
    are the coefficients.
    """
    if nonzero_inds is None:
        # taken from main SHAP package:
        # eliminate one variable with the constraint that all features
        # sum to the output, improves result accuracy significantly
        y = y - X[:, -1] * (fx - expected_value)
        Xw = cp.transpose(
            cp.transpose(X[:, :-1]) - X[:, -1])

        Xw = Xw * cp.sqrt(weights[:, cp.newaxis])
        y = y * cp.sqrt(weights)
        shap_vals = LinearRegression(fit_intercept=False,
                                     output_type='cupy',
                                     handle=handle).fit(Xw, y).coef_

    else:
        # mathematically the same as above, but we need to use the indexes
        # from nonzero_inds and some additional arrays
        # nonzero_inds tells us which cols of X to use
        y = y - X[:, nonzero_inds[-1]] * (fx - expected_value)
        Xw = cp.transpose(
            cp.transpose(X[:, nonzero_inds[:-1]]) - X[:, nonzero_inds[-1]])

        Xw = Xw * cp.sqrt(weights[:, cp.newaxis])
        y = y * cp.sqrt(weights)

        X_t = LinearRegression(fit_intercept=False,
                               output_type='cupy',
                               handle=handle).fit(Xw, y).coef_

        shap_vals = cp.zeros(X.shape[1] - 1)
        shap_vals[nonzero_inds[:-1]] = X_t

    return shap_vals
