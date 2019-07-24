# Copyright (c) 2019, NVIDIA CORPORATION.
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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import cudf
import numpy as np
from numba import cuda
from libc.stdint cimport uintptr_t
from cuml.utils import input_to_dev_array, get_dev_array_ptr, numba_utils
from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle

cdef extern from "holtwinters/holtwinters_params.h" namespace "ML":
    enum SeasonalType:
        ADDITIVE
        MULTIPLICATIVE

cdef extern from "holtwinters/holtwinters.h" namespace "ML::HoltWinters":
    cdef void buffer_size(
        int n, int batch_size, int frequency,
        int *start_leveltrend_len, int *start_season_len,
        int *components_len, int *error_len,
        int *leveltrend_coef_shift, int *season_coef_shift) except +

    cdef void fit(
        cumlHandle &handle, int n, int batch_size,
        int frequency, int start_periods, SeasonalType seasonal,
        float *data, float *level_ptr, float *trend_ptr,
        float *season_ptr, float *SSE_error_ptr) except +
    cdef void fit(
        cumlHandle &handle, int n, int batch_size,
        int frequency, int start_periods, SeasonalType seasonal,
        double *data, double *level_ptr, double *trend_ptr,
        double *season_ptr, double *SSE_error_ptr) except +

    cdef void predict(
        cumlHandle &handle, int n, int batch_size, int frequency,
        int h, SeasonalType seasonal, float *level_ptr,
        float *trend_ptr, float *season_ptr, float *forecast_ptr) except +
    cdef void predict(
        cumlHandle &handle, int n, int batch_size, int frequency,
        int h, SeasonalType seasonal, double *level_ptr,
        double *trend_ptr, double *season_ptr, double *forecast_ptr) except +


class HoltWinters(Base):

    def __init__(self, batch_size=1, freq_season=2,
                 season_type="ADDITIVE", start_periods=2,
                 handle=None):

        super(HoltWinters, self).__init__(handle)

        # Total number of Time Series for forecasting
        if type(batch_size) != int:
            raise TypeError("Type of batch_size must be int. Given: " +
                            type(batch_size))
        if batch_size <= 0:
            raise ValueError("Batch size must be at least 1. Given: " +
                             str(batch_size))
        self.batch_size = batch_size

        # Season length in the time series
        if type(freq_season) != int:
            raise TypeError("Type of freq_season must be int. Given: " +
                            type(freq_season))
        if freq_season < 2:
            raise ValueError("Frequency must be >= 2. Given: " +
                             str(freq_season))
        self.frequency = freq_season

        # whether to perform additive or multiplicative STL decomposition
        if season_type == "ADDITIVE":
            self.season_type = season_type
            self._cpp_stype = ADDITIVE
        elif season_type == "MULTIPLICATIVE":
            self.season_type = season_type
            self._cpp_stype = MULTIPLICATIVE
        else:
            raise ValueError("Season type must be either "
                             "\"ADDITIVE\" or \"MULTIPLICATIVE\"")

        # number of seasons to be used for seasonal seed values
        if type(start_periods) != int:
            raise TypeError("Type of start_periods must be int. Given: " +
                            type(start_periods))
        if start_periods < 2:
            raise ValueError("Start Periods must be >= 2. Given: " +
                             str(start_periods))
        if freq_season < start_periods:
            raise ValueError("Frequency (" + str(freq_season) +
                             ") cannot be less than start_periods (" +
                             str(start_periods) + ").")
        self.start_periods = start_periods

        self.forecasted_points = []  # list for final forecast output
        self.level = []  # list for level values for each time series in batch
        self.trend = []  # list for trend values for each time series in batch
        self.season = []  # list for season values for each series in batch
        self.SSE = []     # SSE for all time series in batch
        self.fit_executed_flag = False
        self.h = 0

    def _check_dims(self, ts_input, is_cudf=False):
        err_mess = ("HoltWinters initialized with " + str(self.batch_size) +
                    " time series, but data has dimension ")
        if len(ts_input.shape) == 1:
            self.n = ts_input.shape[0]
            if self.batch_size != 1:
                raise ValueError(err_mess + "1.")
            if(is_cudf):
                mod_ts_input = ts_input.as_gpu_matrix()
            else:
                mod_ts_input = ts_input
        elif len(ts_input.shape) == 2:
            if(is_cudf):
                d1 = ts_input.shape[0]
                d2 = ts_input.shape[1]
                mod_ts_input = ts_input.as_gpu_matrix()\
                    .reshape((d1*d2,))
            else:
                d1 = ts_input.shape[1]
                d2 = ts_input.shape[0]
                mod_ts_input = ts_input.ravel()
            self.n = d1
            if self.batch_size != d2:
                raise ValueError(err_mess + str(d2))
        else:
            raise ValueError("Data input must have 1 or 2 dimensions.")
        return mod_ts_input

    def fit(self, ts_input):
        if isinstance(ts_input, cudf.DataFrame):
            ts_input = self._check_dims(ts_input, True)
        elif cuda.is_cuda_array(ts_input):
            try:
                import cupy as cp
                ts_input = self._check_dims(ts_input)
            except Exception:
                ts_input = cuda.as_cuda_array(ts_input)
                ts_input = ts_input.copy_to_host()
        if isinstance(ts_input, np.ndarray):
            ts_input = self._check_dims(ts_input)
        if self.n < self.start_periods*self.frequency:
            raise ValueError("Length of time series (" + str(self.n) +
                             ") must be at least freq*start_periods (" +
                             str(self.start_periods*self.frequency) + ").")
        if self.n <= 0:
            raise ValueError("Time series must contain at least 1 value."
                             " Given: " + str(self.n))

        cdef uintptr_t input_ptr
        cdef int leveltrend_seed_len, season_seed_len, components_len
        cdef int leveltrend_coef_offset, season_coef_offset
        cdef int error_len

        X_m, input_ptr, _, _, self.dtype = \
            input_to_dev_array(ts_input, order='C')

        buffer_size(<int> self.n, <int> self.batch_size,
                    <int> self.frequency,
                    <int*> &leveltrend_seed_len,
                    <int*> &season_seed_len,
                    <int*> &components_len,
                    <int*> &leveltrend_coef_offset,
                    <int*> &season_coef_offset,
                    <int*> &error_len)

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()
        cdef uintptr_t level_ptr, trend_ptr, season_ptr, SSE_ptr

        self.level = numba_utils.zeros(components_len, dtype=self.dtype)
        self.trend = numba_utils.zeros(components_len, dtype=self.dtype)
        self.season = numba_utils.zeros(components_len, dtype=self.dtype)
        self.SSE = numba_utils.zeros(self.batch_size, dtype=self.dtype)
        level_ptr = get_dev_array_ptr(self.level)
        trend_ptr = get_dev_array_ptr(self.trend)
        season_ptr = get_dev_array_ptr(self.season)
        SSE_ptr = get_dev_array_ptr(self.SSE)

        if self.dtype == np.float32:
            fit(handle_[0], <int> self.n, <int> self.batch_size,
                <int> self.frequency, <int> self.start_periods,
                <SeasonalType> self._cpp_stype,
                <float*> input_ptr, <float*> level_ptr,
                <float*> trend_ptr, <float*> season_ptr,
                <float*> SSE_ptr)

        elif self.dtype == np.float64:
            fit(handle_[0], <int> self.n, <int> self.batch_size,
                <int> self.frequency, <int> self.start_periods,
                <SeasonalType> self._cpp_stype,
                <double*> input_ptr, <double*> level_ptr,
                <double*> trend_ptr, <double*> season_ptr,
                <double*> SSE_ptr)

        else:
            raise TypeError("HoltWinters supports only float32"
                            " and float64 input, but input type "
                            + str(self.dtype) + " passed.")
        self.handle.sync()
        self.fit_executed_flag = True
        del(X_m)
        return self

    def forecast(self, h=1, index=None):
        cdef uintptr_t forecast_ptr, level_ptr, trend_ptr, season_ptr
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        if type(h) != int or (type(index) != int and index is not None):
            raise TypeError("Input arguments must be of type int."
                            "Index has type: " + str(type(index))
                            + "\nh has type: " + str(type(h)))

        if self.fit_executed_flag:
            if h <= 0:
                raise ValueError("h must be > 0. Currently: " + str(h))

            if h > self.h:
                self.h = h
                self.forecasted_points = numba_utils.zeros(self.batch_size*h,
                                                           dtype=self.dtype)
                forecast_ptr = get_dev_array_ptr(self.forecasted_points)
                level_ptr = get_dev_array_ptr(self.level)
                trend_ptr = get_dev_array_ptr(self.trend)
                season_ptr = get_dev_array_ptr(self.season)

                if self.dtype == np.float32:
                    predict(handle_[0], <int> self.n,
                            <int> self.batch_size,
                            <int> self.frequency,
                            <int> h,
                            <SeasonalType> self._cpp_stype,
                            <float*> level_ptr,
                            <float*> trend_ptr,
                            <float*> season_ptr,
                            <float*> forecast_ptr)
                elif self.dtype == np.float64:
                    predict(handle_[0], <int> self.n,
                            <int> self.batch_size,
                            <int> self.frequency, <int> h,
                            <SeasonalType> self._cpp_stype,
                            <double*> level_ptr,
                            <double*> trend_ptr,
                            <double*> season_ptr,
                            <double*> forecast_ptr)
                self.forecasted_points =\
                    self.forecasted_points.reshape((self.batch_size, h),
                                                   order='F')
                self.handle.sync()

            if index is None:
                if self.batch_size == 1:
                    return cudf.Series(
                        self.forecasted_points.ravel(order='F')[:h])
                else:
                    return cudf.DataFrame.from_gpu_matrix(
                        self.forecasted_points[:, :h].T)
            else:
                if index < 0 or index >= self.batch_size:
                    raise IndexError("Index input: " + str(index) +
                                     " outside of range [0, " +
                                     str(self.batch_size) + "]")
                return cudf.Series(self.forecasted_points[index, :h])
        else:
            raise ValueError("Fit() the model before forecast()")

    def score(self, index=None):
        if self.fit_executed_flag:
            if index is None:
                return cudf.Series(self.SSE)
            elif index < 0 or index >= self.batch_size:
                raise IndexError("Index input: " + str(index) + " outside of "
                                 "range [0, " + str(self.batch_size) + "]")
            else:
                return self.SSE[index]
        else:
            raise ValueError("Fit() the model before score()")

    def get_level(self, index=None):
        if self.fit_executed_flag:
            if index is None:
                return cudf.Series(self.level)
            elif index < 0 or index >= self.batch_size:
                raise IndexError("Index input: " + str(index) + " outside of "
                                 "range [0, " + str(self.batch_size) + "]")
            else:
                return self.level[index]
        else:
            raise ValueError("Fit() the model to get level values")

    def get_trend(self, index=None):
        if self.fit_executed_flag:
            if index is None:
                return cudf.Series(self.trend)
            elif index < 0 or index >= self.batch_size:
                raise IndexError("Index input: " + str(index) + " outside of "
                                 "range [0, " + str(self.batch_size) + "]")
            else:
                return self.trend[index]
        else:
            raise ValueError("Fit() the model to get trend values")

    def get_season(self, index=None):
        if self.fit_executed_flag:
            if index is None:
                return cudf.Series(self.season)
            elif index < 0 or index >= self.batch_size:
                raise IndexError("Index input: " + str(index) + " outside of "
                                 "range [0, " + str(self.batch_size) + "]")
            else:
                return self.season[index]
        else:
            raise ValueError("Fit() the model to get season values")
