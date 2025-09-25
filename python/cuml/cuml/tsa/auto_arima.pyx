#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

# distutils: language = c++

import itertools
import typing

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from libcpp.vector cimport vector

import cupy as cp
import numpy as np

import cuml.internals
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.internals import logger
from cuml.internals.array import CumlArray
from cuml.internals.base import Base

from pylibraft.common.handle cimport handle_t

from pylibraft.common.handle import Handle

from cuml.common import input_to_cuml_array, using_output_type
from cuml.tsa.arima import ARIMA
from cuml.tsa.seasonality import seas_test
from cuml.tsa.stationarity import kpss_test

# TODO:
# - Box-Cox transformations? (parameter lambda)
# - Would a "one-fits-all" method be useful?


cdef extern from "cuml/tsa/auto_arima.h" namespace "ML" nogil:
    int divide_by_mask_build_index(const handle_t& handle, const bool* mask,
                                   int* index, int batch_size) except +

    void divide_by_mask_execute(const handle_t& handle, const float* d_in,
                                const bool* mask, const int* index,
                                float* d_out0, float* d_out1, int batch_size,
                                int n_obs) except +
    void divide_by_mask_execute(const handle_t& handle, const double* d_in,
                                const bool* mask, const int* index,
                                double* d_out0, double* d_out1,
                                int batch_size, int n_obs) except +
    void divide_by_mask_execute(const handle_t& handle, const int* d_in,
                                const bool* mask, const int* index,
                                int* d_out0, int* d_out1, int batch_size,
                                int n_obs) except +

    void divide_by_min_build_index(const handle_t& handle,
                                   const float* d_matrix, int* d_batch,
                                   int* d_index, int* h_size,
                                   int batch_size, int n_sub) except +
    void divide_by_min_build_index(const handle_t& handle,
                                   const double* d_matrix, int* d_batch,
                                   int* d_index, int* h_size,
                                   int batch_size, int n_sub) except +

    void divide_by_min_execute(const handle_t& handle, const float* d_in,
                               const int* d_batch, const int* d_index,
                               float** hd_out, int batch_size, int n_sub,
                               int n_obs) except +
    void divide_by_min_execute(const handle_t& handle, const double* d_in,
                               const int* d_batch, const int* d_index,
                               double** hd_out, int batch_size, int n_sub,
                               int n_obs) except +
    void divide_by_min_execute(const handle_t& handle, const int* d_in,
                               const int* d_batch, const int* d_index,
                               int** hd_out, int batch_size, int n_sub,
                               int n_obs) except +

    void cpp_build_division_map "ML::build_division_map" (
        const handle_t& handle, const int* const* hd_id, const int* h_size,
        int* d_id_to_pos, int* d_id_to_model, int batch_size, int n_sub) except +

    void cpp_merge_series "ML::merge_series" (
        const handle_t& handle, const float* const* hd_in,
        const int* d_id_to_pos, const int* d_id_to_sub, float* d_out,
        int batch_size, int n_sub, int n_obs) except +
    void cpp_merge_series "ML::merge_series" (
        const handle_t& handle, const double* const* hd_in,
        const int* d_id_to_pos, const int* d_id_to_sub, double* d_out,
        int batch_size, int n_sub, int n_obs) except +

cdef extern from "cuml/tsa/batched_arima.hpp" namespace "ML" nogil:
    bool detect_missing(
        handle_t& handle, const double* d_y, int n_elem) except +

tests_map = {
    "kpss": kpss_test,
    "seas": seas_test,
}


class AutoARIMA(Base):
    """
    Implements a batched auto-ARIMA model for in- and out-of-sample
    times-series prediction.

    This interface offers a highly customizable search, with functionality
    similar to the `forecast` and `fable` packages in R. It provides an
    abstraction around the underlying ARIMA models to predict and forecast as
    if using a single model.

    Parameters
    ----------

    endog : dataframe or array-like (device or host)
        The time series data, assumed to have each time series in columns.
        Acceptable formats: cuDF DataFrame, cuDF Series, NumPy ndarray,
        Numba device ndarray, cuda array interface compliant array like CuPy.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    simple_differencing: bool or int, default=True
        If True, the data is differenced before being passed to the Kalman
        filter. If False, differencing is part of the state-space model.
        See additional notes in the ARIMA docs
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    convert_dtype : boolean
        When set to True, the model will automatically convert the inputs to
        np.float64.

    Notes
    -----

    The interface was influenced by the R `fable` package:
    See https://fable.tidyverts.org/reference/ARIMA.html

    References
    ----------

    A useful (though outdated) reference is the paper:

    .. [1] Rob J. Hyndman, Yeasmin Khandakar, 2008. "Automatic Time Series
        Forecasting: The 'forecast' Package for R", Journal of Statistical
        Software 27

    Examples
    --------

    .. code-block:: python

            from cuml.tsa.auto_arima import AutoARIMA

            model = AutoARIMA(y)
            model.search(s=12, d=(0, 1), D=(0, 1), p=(0, 2, 4), q=(0, 2, 4),
                         P=range(2), Q=range(2), method="css", truncate=100)
            model.fit(method="css-ml")
            fc = model.forecast(20)


    """

    d_y = CumlArrayDescriptor()

    def __init__(self,
                 endog,
                 *,
                 handle=None,
                 simple_differencing=True,
                 verbose=False,
                 output_type=None,
                 convert_dtype=True):
        # Initialize base class
        super().__init__(handle=handle,
                         verbose=verbose,
                         output_type=output_type)
        self._set_base_attributes(output_type=endog)

        # Get device array. Float64 only for now.
        self.d_y, self.n_obs, self.batch_size, self.dtype \
            = input_to_cuml_array(
                endog, check_dtype=np.float64,
                convert_to_dtype=(np.float64 if convert_dtype else None))

        self.simple_differencing = simple_differencing

        self._initial_calc()

    @cuml.internals.api_base_return_any_skipall
    def _initial_calc(self):
        cdef uintptr_t d_y_ptr = self.d_y.ptr
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        # Detect missing observations
        missing = detect_missing(handle_[0], <double*> d_y_ptr,
                                 <int> self.batch_size * self.n_obs)

        if missing:
            raise ValueError(
                "Missing observations are not supported in AutoARIMA yet")

    @cuml.internals.api_return_any()
    def search(self,
               s=None,
               d=range(3),
               D=range(2),
               p=range(1, 4),
               q=range(1, 4),
               P=range(3),
               Q=range(3),
               fit_intercept="auto",
               ic="aicc",
               test="kpss",
               seasonal_test="seas",
               h: float = 1e-8,
               maxiter: int = 1000,
               method="auto",
               truncate: int = 0):
        """Searches through the specified model space and associates each
        series to the most appropriate model.

        Parameters
        ----------
        s : int
            Seasonal period. None or 0 for non-seasonal time series
        d : int, sequence or generator
            Possible values for d (simple difference)
        D : int, sequence or generator
            Possible values for D (seasonal difference)
        p : int, sequence or generator
            Possible values for p (AR order)
        q : int, sequence or generator
            Possible values for q (MA order)
        P : int, sequence or generator
            Possible values for P (seasonal AR order)
        Q : int, sequence or generator
            Possible values for Q (seasonal MA order)
        fit_intercept : int, sequence, generator or "auto"
            Whether to fit an intercept. "auto" chooses based on the model
            parameters: it uses an incercept iff d + D <= 1
        ic : str
            Which information criterion to use for the model selection.
            Currently supported: AIC, AICc, BIC
        test : str
            Which stationarity test to use to choose d.
            Currently supported: KPSS
        seasonal_test : str
            Which seasonality test to use to choose D.
            Currently supported: seas
        h : float
            Finite-differencing step size used to compute gradients in ARIMA
        maxiter : int
            Maximum number of iterations of L-BFGS-B
        method : str
            Estimation method - "auto", "css", "css-ml" or "ml".
            CSS uses a fast sum-of-squares approximation.
            ML estimates the log-likelihood with statespace methods.
            CSS-ML starts with CSS and refines with ML.
            "auto" will use CSS for long seasonal time series, ML otherwise.
        truncate : int
            When using CSS, start the sum of squares after a given number of
            observations for better performance. Recommended for long time
            series when truncating doesn't lose too much information.
        """
        # Notes:
        #  - We iteratively divide the dataset as we decide parameters, so
        #    it's important to make sure that we don't keep the unused arrays
        #    alive, so they can get garbage-collected.
        #  - As we divide the dataset, we also keep track of the original
        #    index of each series in the batch, to construct the final map at
        #    the end.

        # Parse input parameters
        ic = ic.lower()
        test = test.lower()
        seasonal_test = seasonal_test.lower()
        if s is None or s == 1:  # R users might use s=1 for non-seasonal data
            s = 0
        if method == "auto":
            method = "css" if self.n_obs >= 100 and s >= 4 else "ml"

        # Original index
        d_index, *_ = input_to_cuml_array(np.r_[:self.batch_size],
                                          convert_to_dtype=np.int32)

        #
        # Choose the hyper-parameter D
        #
        logger.info("Deciding D...")
        D_options = _parse_sequence("D", D, 0, 1)
        if not s:
            # Non-seasonal -> D=0
            data_D = {0: (self.d_y, d_index)}
        elif len(D_options) == 1:
            # D is specified by the user
            data_D = {D_options[0]: (self.d_y, d_index)}
        else:
            # D is chosen with a seasonal differencing test
            if seasonal_test not in tests_map:
                raise ValueError("Unknown seasonal diff test: {}"
                                 .format(seasonal_test))

            with using_output_type("cupy"):
                mask_cp = tests_map[seasonal_test](self.d_y, s)

            mask = input_to_cuml_array(mask_cp)[0]
            del mask_cp
            data_D = {}
            out0, index0, out1, index1 = _divide_by_mask(self.d_y, mask,
                                                         d_index)
            if out0 is not None:
                data_D[0] = (out0, index0)
            if out1 is not None:
                data_D[1] = (out1, index1)
            del mask, out0, index0, out1, index1

        #
        # Choose the hyper-parameter d
        #
        logger.info("Deciding d...")
        data_dD = {}
        for D_ in data_D:
            d_options = _parse_sequence("d", d, 0, 2 - D_)
            if len(d_options) == 1:
                # d is specified by the user
                data_dD[(d_options[0], D_)] = data_D[D_]
            else:
                # d is decided with stationarity tests
                if test not in tests_map:
                    raise ValueError("Unknown stationarity test: {}"
                                     .format(test))
                data_temp, id_temp = data_D[D_]
                for d_ in d_options[:-1]:
                    mask_cp = tests_map[test](data_temp.to_output("cupy"),
                                              d_, D_, s)
                    mask = input_to_cuml_array(mask_cp)[0]
                    del mask_cp
                    out0, index0, out1, index1 \
                        = _divide_by_mask(data_temp, mask, id_temp)
                    if out1 is not None:
                        data_dD[(d_, D_)] = (out1, index1)
                    if out0 is not None:
                        (data_temp, id_temp) = (out0, index0)
                    else:
                        break
                else:  # (when the for loop reaches its end naturally)
                    # The remaining series are assigned the max possible d
                    data_dD[(d_options[-1], D_)] = (data_temp, id_temp)
                del data_temp, id_temp, mask, out0, index0, out1, index1
        del data_D

        #
        # Choose the hyper-parameters p, q, P, Q, k
        #
        logger.info("Deciding p, q, P, Q, k...")
        p_options = _parse_sequence("p", p, 0, s - 1 if s else 4)
        q_options = _parse_sequence("q", q, 0, s - 1 if s else 4)
        P_options = _parse_sequence("P", P, 0, 4 if s else 0)
        Q_options = _parse_sequence("Q", Q, 0, 4 if s else 0)
        self.models = []
        id_tracker = []
        for (d_, D_) in data_dD:
            data_temp, id_temp = data_dD[(d_, D_)]
            batch_size = data_temp.shape[1] if len(data_temp.shape) > 1 else 1

            k_options = ([1 if d_ + D_ <= 1 else 0] if fit_intercept == "auto"
                         else _parse_sequence("k", fit_intercept, 0, 1))

            # Grid search
            all_ic = []
            all_orders = []
            for p_, q_, P_, Q_, k_ in itertools.product(p_options, q_options,
                                                        P_options, Q_options,
                                                        k_options):
                if p_ + q_ + P_ + Q_ + k_ == 0:
                    continue
                s_ = s if (P_ + D_ + Q_) else 0
                model = ARIMA(endog=data_temp.to_output("cupy"),
                              order=(p_, d_, q_),
                              seasonal_order=(P_, D_, Q_, s_),
                              fit_intercept=k_,
                              handle=self.handle,
                              simple_differencing=self.simple_differencing,
                              output_type="cupy")
                logger.debug("Fitting {} ({})".format(model, method))
                model.fit(h=h, maxiter=maxiter, method=method,
                          truncate=truncate)
                all_ic.append(model._ic(ic))
                all_orders.append((p_, q_, P_, Q_, s_, k_))
                del model

            # Organize the results into a matrix
            n_models = len(all_orders)
            ic_matrix, *_ = input_to_cuml_array(
                cp.concatenate([ic_arr.to_output('cupy').reshape(batch_size, 1)
                                for ic_arr in all_ic], 1))

            # Divide the batch, choosing the best model for each series
            sub_batches, sub_id = _divide_by_min(data_temp, ic_matrix, id_temp)
            for i in range(n_models):
                if sub_batches[i] is None:
                    continue
                p_, q_, P_, Q_, s_, k_ = all_orders[i]
                self.models.append(
                    ARIMA(sub_batches[i].to_output("cupy"), order=(p_, d_, q_),
                          seasonal_order=(P_, D_, Q_, s_), fit_intercept=k_,
                          handle=self.handle, output_type="cupy",
                          simple_differencing=self.simple_differencing))
                id_tracker.append(sub_id[i])

            del all_ic, all_orders, ic_matrix, sub_batches, sub_id

        # Build a map to match each series to its model and position in the
        # sub-batch
        logger.info("Finalizing...")
        self.id_to_model, self.id_to_pos = _build_division_map(id_tracker,
                                                               self.batch_size)

    @cuml.internals.api_base_return_any_skipall
    def fit(self,
            h: float = 1e-8,
            maxiter: int = 1000,
            method="ml",
            truncate: int = 0):
        """Fits the selected models for their respective series

        Parameters
        ----------
        h : float
            Finite-differencing step size used to compute gradients in ARIMA
        maxiter : int
            Maximum number of iterations of L-BFGS-B
        method : str
            Estimation method - "css", "css-ml" or "ml".
            CSS uses a fast sum-of-squares approximation.
            ML estimates the log-likelihood with statespace methods.
            CSS-ML starts with CSS and refines with ML.
        truncate : int
            When using CSS, start the sum of squares after a given number of
            observations for better performance (but often a worse fit)
        """
        for model in self.models:
            logger.debug("Fitting {} ({})".format(model, method))
            model.fit(h=h, maxiter=maxiter, method=method, truncate=truncate)

    @cuml.internals.api_base_return_generic_skipall
    def predict(
        self,
        start=0,
        end=None,
        level=None
    ) -> typing.Union[CumlArray, typing.Tuple[CumlArray, CumlArray,
                                              CumlArray]]:
        """Compute in-sample and/or out-of-sample prediction for each series

        Parameters
        ----------
        start: int
            Index where to start the predictions (0 <= start <= num_samples)
        end:
            Index where to end the predictions, excluded (end > start)
        level: float or None (default = None)
            Confidence level for prediction intervals, or None to return only
            the point forecasts. 0 < level < 1

        Returns
        -------
        y_p : array-like (device)
            Predictions. Shape = (end - start, batch_size)
        lower: array-like (device) (optional)
            Lower limit of the prediction interval if level != None
            Shape = (end - start, batch_size)
        upper: array-like (device) (optional)
            Upper limit of the prediction interval if level != None
            Shape = (end - start, batch_size)
        """
        # Compute predictions for each model
        pred_list = []
        lower_list = []
        upper_list = []
        for model in self.models:
            if level is None:
                pred, *_ = input_to_cuml_array(model.predict(start, end))
                pred_list.append(pred)
            else:
                pred, low, upp = model.predict(start, end, level=level)
                pred_list.append(input_to_cuml_array(pred)[0])
                lower_list.append(input_to_cuml_array(low)[0])
                upper_list.append(input_to_cuml_array(upp)[0])

        # Put all the predictions together
        y_p = _merge_series(pred_list, self.id_to_model, self.id_to_pos,
                            self.batch_size)
        if level is not None:
            lower = _merge_series(lower_list, self.id_to_model, self.id_to_pos,
                                  self.batch_size)
            upper = _merge_series(upper_list, self.id_to_model, self.id_to_pos,
                                  self.batch_size)

        # Return the results
        if level is None:
            return y_p
        else:
            return y_p, lower, upper

    @cuml.internals.api_base_return_generic_skipall
    def forecast(self,
                 nsteps: int,
                 level=None) -> typing.Union[CumlArray,
                                             typing.Tuple[CumlArray,
                                                          CumlArray,
                                                          CumlArray]]:
        """Forecast `nsteps` into the future.

        Parameters
        ----------
        nsteps : int
            The number of steps to forecast beyond end of the given series
        level: float or None (default = None)
            Confidence level for prediction intervals, or None to return only
            the point forecasts. 0 < level < 1

        Returns
        -------
        y_fc : array-like
               Forecasts. Shape = (nsteps, batch_size)
        lower: array-like (device) (optional)
            Lower limit of the prediction interval if level != None
            Shape = (end - start, batch_size)
        upper: array-like (device) (optional)
            Upper limit of the prediction interval if level != None
            Shape = (end - start, batch_size)
        """
        return self.predict(self.n_obs, self.n_obs + nsteps, level)

    def summary(self):
        """Display a quick summary of the models selected by `search`
        """
        model_list = sorted(self.models, key=lambda model: model.batch_size,
                            reverse=True)
        print("ARIMA models used:", len(model_list))
        for model in model_list:
            print(" -", str(model))


# Helper functions

def _parse_sequence(name, seq_in, min_accepted, max_accepted):
    """Convert a sequence/generator/integer into a sorted list, keeping
    only values within the accepted range
    """
    seq_temp = [seq_in] if type(seq_in) is int else seq_in
    seq_out = sorted(x for x in seq_temp
                     if x >= min_accepted and x <= max_accepted)
    if len(seq_out) == 0:
        raise ValueError("No valid option for {}".format(name))
    else:
        return seq_out


def _divide_by_mask(original, mask, batch_id, handle=None):
    """Divide a given batch into two sub-batches according to a boolean mask

    .. note:: in case the mask contains only False or only True, one sub-batch
        will be the original batch (not a copy!) and the other None

    Parameters
    ----------
    original : CumlArray (float32 or float64)
        Original batch
    mask : CumlArray (bool)
        Boolean mask: False for the 1st sub-batch and True for the second
    batch_id : CumlArray (int)
        Integer array to track the id of each member in the initial batch
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.

    Returns
    -------
    out0 : CumlArray (float32 or float64)
        Sub-batch 0, or None if empty
    batch0_id : CumlArray (int)
        Indices of the members of the sub-batch 0 in the initial batch,
        or None if empty
    out1 : CumlArray (float32 or float64)
        Sub-batch 1, or None if empty
    batch1_id : CumlArray (int)
        Indices of the members of the sub-batch 1 in the initial batch,
        or None if empty
    """
    assert batch_id.dtype == np.int32

    dtype = original.dtype
    n_obs = original.shape[0]
    batch_size = original.shape[1] if len(original.shape) > 1 else 1

    if handle is None:
        handle = Handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

    index = CumlArray.empty(batch_size, np.int32)
    cdef uintptr_t d_index = index.ptr
    cdef uintptr_t d_mask = mask.ptr

    # Compute the index of each series in their new batch
    nb_true = divide_by_mask_build_index(handle_[0],
                                         <bool*> d_mask,
                                         <int*> d_index,
                                         <int> batch_size)

    out0 = CumlArray.empty((n_obs, batch_size - nb_true), dtype)
    out1 = CumlArray.empty((n_obs, nb_true), dtype)

    # Type declarations (can't be in if-else statements)
    cdef uintptr_t d_out0
    cdef uintptr_t d_out1
    cdef uintptr_t d_original = original.ptr
    cdef uintptr_t d_batch0_id
    cdef uintptr_t d_batch1_id
    cdef uintptr_t d_batch_id

    # If the sub-batch 1 is empty
    if nb_true == 0:
        out0 = original
        out1 = None
        batch0_id = batch_id
        batch1_id = None

    # If the sub-batch 0 is empty
    elif nb_true == batch_size:
        out0 = None
        out1 = original
        batch0_id = None
        batch1_id = batch_id

    # If both sub-batches have elements
    else:
        out0 = CumlArray.empty((n_obs, batch_size - nb_true), dtype)
        out1 = CumlArray.empty((n_obs, nb_true), dtype)
        d_out0 = out0.ptr
        d_out1 = out1.ptr

        # Build the two sub-batches
        if dtype == np.float32:
            divide_by_mask_execute(handle_[0],
                                   <float*> d_original,
                                   <bool*> d_mask,
                                   <int*> d_index,
                                   <float*> d_out0,
                                   <float*> d_out1,
                                   <int> batch_size,
                                   <int> n_obs)
        else:
            divide_by_mask_execute(handle_[0],
                                   <double*> d_original,
                                   <bool*> d_mask,
                                   <int*> d_index,
                                   <double*> d_out0,
                                   <double*> d_out1,
                                   <int> batch_size,
                                   <int> n_obs)

        # Also keep track of the original id of the series in the batch
        batch0_id = CumlArray.empty(batch_size - nb_true, np.int32)
        batch1_id = CumlArray.empty(nb_true, np.int32)
        d_batch0_id = batch0_id.ptr
        d_batch1_id = batch1_id.ptr
        d_batch_id = batch_id.ptr

        divide_by_mask_execute(handle_[0],
                               <int*> d_batch_id,
                               <bool*> d_mask,
                               <int*> d_index,
                               <int*> d_batch0_id,
                               <int*> d_batch1_id,
                               <int> batch_size,
                               <int> 1)

    return out0, batch0_id, out1, batch1_id


def _divide_by_min(original, metrics, batch_id, handle=None):
    """Divide a given batch into multiple sub-batches according to the values
    of the given metrics, by selecting the minimum value for each member

    Parameters:
    ----------
    original : CumlArray (float32 or float64)
        Original batch
    metrics : CumlArray (float32 or float64)
        Matrix of shape (batch_size, n_sub) containing the metrics to minimize
    batch_id : CumlArray (int)
        Integer array to track the id of each member in the initial batch
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.

    Returns
    -------
    sub_batches : List[CumlArray] (float32 or float64)
        List of arrays containing each sub-batch, or None if empty
    sub_id : List[CumlArray] (int)
        List of arrays containing the indices of each member in the initial
        batch, or None if empty
    """
    assert batch_id.dtype == np.int32

    dtype = original.dtype
    n_obs = original.shape[0]
    n_sub = metrics.shape[1]
    batch_size = original.shape[1] if len(original.shape) > 1 else 1

    if handle is None:
        handle = Handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

    batch_buffer = CumlArray.empty(batch_size, np.int32)
    index_buffer = CumlArray.empty(batch_size, np.int32)
    cdef vector[int] size_buffer
    size_buffer.resize(n_sub)

    cdef uintptr_t d_metrics = metrics.ptr
    cdef uintptr_t d_batch = batch_buffer.ptr
    cdef uintptr_t d_index = index_buffer.ptr

    # Compute which sub-batch each series belongs to, its position in
    # the sub-batch, and the size of each sub-batch
    if dtype == np.float32:
        divide_by_min_build_index(handle_[0],
                                  <float*> d_metrics,
                                  <int*> d_batch,
                                  <int*> d_index,
                                  <int*> size_buffer.data(),
                                  <int> batch_size,
                                  <int> n_sub)
    else:
        divide_by_min_build_index(handle_[0],
                                  <double*> d_metrics,
                                  <int*> d_batch,
                                  <int*> d_index,
                                  <int*> size_buffer.data(),
                                  <int> batch_size,
                                  <int> n_sub)

    # Build a list of cuML arrays for the sub-batches and a vector of pointers
    # to be passed to the next C++ step
    sub_batches = [CumlArray.empty((n_obs, s), dtype) if s else None
                   for s in size_buffer]
    cdef vector[uintptr_t] sub_ptr
    sub_ptr.resize(n_sub)
    for i in range(n_sub):
        if size_buffer[i]:
            sub_ptr[i] = <uintptr_t> sub_batches[i].ptr
        else:
            sub_ptr[i] = <uintptr_t> NULL

    # Execute the batch sub-division
    cdef uintptr_t d_original = original.ptr
    if dtype == np.float32:
        divide_by_min_execute(handle_[0],
                              <float*> d_original,
                              <int*> d_batch,
                              <int*> d_index,
                              <float**> sub_ptr.data(),
                              <int> batch_size,
                              <int> n_sub,
                              <int> n_obs)
    else:
        divide_by_min_execute(handle_[0],
                              <double*> d_original,
                              <int*> d_batch,
                              <int*> d_index,
                              <double**> sub_ptr.data(),
                              <int> batch_size,
                              <int> n_sub,
                              <int> n_obs)

    # Keep track of the id of the series if requested
    cdef vector[uintptr_t] id_ptr
    sub_id = [CumlArray.empty(s, np.int32) if s else None
              for s in size_buffer]
    id_ptr.resize(n_sub)
    for i in range(n_sub):
        if size_buffer[i]:
            id_ptr[i] = <uintptr_t> sub_id[i].ptr
        else:
            id_ptr[i] = <uintptr_t> NULL

    cdef uintptr_t d_batch_id = batch_id.ptr
    divide_by_min_execute(handle_[0],
                          <int*> d_batch_id,
                          <int*> d_batch,
                          <int*> d_index,
                          <int**> id_ptr.data(),
                          <int> batch_size,
                          <int> n_sub,
                          <int> 1)

    return sub_batches, sub_id


def _build_division_map(id_tracker, batch_size, handle=None):
    """Build a map to associate each batch member with a model and index in
    the associated sub-batch

    Parameters
    ----------
    id_tracker : List[CumlArray] (int)
        List of the index arrays of each sub-batch
    batch_size : int
        Size of the initial batch

    Returns
    -------
    id_to_model : CumlArray (int)
        Associates each batch member with a model
    id_to_pos : CumlArray (int)
        Position of each member in the respective sub-batch
    """
    if handle is None:
        handle = Handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

    n_sub = len(id_tracker)

    id_to_pos = CumlArray.empty(batch_size, np.int32)
    id_to_model = CumlArray.empty(batch_size, np.int32)

    cdef vector[uintptr_t] id_ptr
    cdef vector[int] size_vec
    id_ptr.resize(n_sub)
    size_vec.resize(n_sub)
    for i in range(n_sub):
        id_ptr[i] = id_tracker[i].ptr
        size_vec[i] = len(id_tracker[i])

    cdef uintptr_t hd_id = <uintptr_t> id_ptr.data()
    cdef uintptr_t h_size = <uintptr_t> size_vec.data()
    cdef uintptr_t d_id_to_pos = id_to_pos.ptr
    cdef uintptr_t d_id_to_model = id_to_model.ptr

    cpp_build_division_map(handle_[0],
                           <const int**> hd_id,
                           <int*> h_size,
                           <int*> d_id_to_pos,
                           <int*> d_id_to_model,
                           <int> batch_size,
                           <int> n_sub)

    return id_to_model, id_to_pos


def _merge_series(data_in, id_to_sub, id_to_pos, batch_size, handle=None):
    """Merge multiple sub-batches into one batch according to the maps that
    associate each id in the unique batch to a sub-batch and a position in
    this sub-batch.

    Parameters
    ----------
    data_in : List[CumlArray] (float32 or float64)
        List of sub-batches to merge
    id_to_model : CumlArray (int)
        Associates each member of the batch with a sub-batch
    id_to_pos : CumlArray (int)
        Position of each member of the batch in its respective sub-batch
    batch_size : int
        Size of the initial batch

    Returns
    -------
    data_out : CumlArray (float32 or float64)
        Merged batch
    """
    dtype = data_in[0].dtype
    n_obs = data_in[0].shape[0]
    n_sub = len(data_in)

    if handle is None:
        handle = Handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

    cdef vector[uintptr_t] in_ptr
    in_ptr.resize(n_sub)
    for i in range(n_sub):
        in_ptr[i] = data_in[i].ptr

    data_out = CumlArray.empty((n_obs, batch_size), dtype)

    cdef uintptr_t hd_in = <uintptr_t> in_ptr.data()
    cdef uintptr_t d_id_to_pos = id_to_pos.ptr
    cdef uintptr_t d_id_to_sub = id_to_sub.ptr
    cdef uintptr_t d_out = data_out.ptr

    if dtype == np.float32:
        cpp_merge_series(handle_[0],
                         <const float**> hd_in,
                         <int*> d_id_to_pos,
                         <int*> d_id_to_sub,
                         <float*> d_out,
                         <int> batch_size,
                         <int> n_sub,
                         <int> n_obs)
    else:
        cpp_merge_series(handle_[0],
                         <const double**> hd_in,
                         <int*> d_id_to_pos,
                         <int*> d_id_to_sub,
                         <double*> d_out,
                         <int> batch_size,
                         <int> n_sub,
                         <int> n_obs)

    return data_out
