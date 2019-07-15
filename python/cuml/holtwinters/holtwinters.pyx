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
from cuml.utils import input_to_dev_array
from cuml.common.base import Base

cdef extern from "holtwinters/HoltWinters.hpp" namespace "ML":
    enum SeasonalType:
        ADDITIVE
        MULTIPLICATIVE

    cdef void HoltWintersFitPredict[Dtype](
        int n, int batch_size, int frequency, int h,
        int start_periods, SeasonalType seasonal,
        Dtype * data, Dtype * alpha_ptr, Dtype * beta_ptr,
        Dtype * gamma_ptr, Dtype * SSE_error_ptr,
        Dtype * forecast_ptr) except +


class HoltWinters(Base):

    def __init__(self, batch_size=1, freq_season=2,
                 season_type="ADDITIVE", start_periods=2):

        # Total number of Time Series for forecasting
        if type(batch_size) != int:
            raise TypeError("Type of batch_size must be int. Given: " +
                            type(batch_size))
        self.batch_size = batch_size

        # Season length in the time series
        if type(freq_season) != int:
            raise TypeError("Type of freq_season must be int. Given: " +
                            type(freq_season))
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

        self.forecasted_points = []  # list for final forecast output
        self.alpha = []  # list for alpha values for each time series in batch
        self.beta = []   # list for beta values for each time series in batch
        self.gamma = []  # list for gamma values for each time series in batch
        self.SSE_error = []          # SSE Error for all time series in batch
        self.h = 50      # Default number of points to forecast in future
        self.fit_executed_flag = False

        if type(start_periods) != int:
            raise TypeError("Type of start_periods must be int. Given: " +
                            type(start_periods))

        if freq_season < start_periods:
            raise ValueError("Frequency (" + str(freq_season) +
                             ") cannot be less than start_periods (" +
                             str(start_periods) + ").")
        else:
            # number of seasons to be used for seasonal seed values
            self.start_periods = start_periods

    def fit(self, ts_input, pointsToForecast=50):
        if type(pointsToForecast) != int:
            raise TypeError("pointToForecast must be of type int. Given: "
                            + str(type(pointsToForecast)))

        self.h = pointsToForecast

        cdef uintptr_t input_ptr

        if isinstance(ts_input, cudf.DataFrame):
            if len(ts_input.shape) == 1:
                self.n = ts_input.shape[0]
                if self.batch_size != 1:
                    raise ValueError("HoltWinters initialized with " +
                                     str(self.batch_size) + " time series "
                                     "but data input has size 1.")
                ts_input = ts_input.as_gpu_matrix()
            elif len(ts_input.shape) == 2:
                self.n = ts_input.shape[0]
                if self.batch_size != ts_input.shape[1]:
                    raise ValueError("HoltWinters initialized with " +
                                     str(self.batch_size) + " time series "
                                     "but data input has " +
                                     str(ts_input.shape[1]) + " series.")
                ts_input = ts_input.as_gpu_matrix()\
                    .reshape((self.n*self.batch_size,))
            else:
                raise ValueError("Data input must have 1 or 2 dimensions.")
        elif cuda.is_cuda_array(ts_input):
            try:
                import cupy as cp
                if len(ts_input.shape) == 2:
                    self.n = ts_input.shape[1]
                    if ts_input.shape[0] != self.batch_size:
                        raise ValueError("HoltWinters initialized with " +
                                         str(self.batch_size) + " time "
                                         "series, but data input has " +
                                         str(ts_input.shape[0]) + " series.")
                    ts_input = ts_input.ravel()
                elif len(ts_input.shape) == 1:
                    self.n = ts_input.shape[0]
                    if self.batch_size != 1:
                        raise ValueError("HoltWinters initialized with " +
                                         str(self.batch_size) + " time "
                                         "series but data has size 1.")
                else:
                    raise ValueError("Data must have 1 or 2 dimensions.")
            except Exception:
                ts_input = cuda.as_cuda_array(ts_input)
                ts_input = ts_input.copy_to_host()
        if isinstance(ts_input, np.ndarray):
            if len(ts_input.shape) == 2:
                self.n = ts_input.shape[1]
                if ts_input.shape[0] != self.batch_size:
                    raise ValueError("HoltWinters initialized with " +
                                     str(self.batch_size) + " time "
                                     "series, but data input has " +
                                     str(ts_input.shape[0]) + " series.")
                ts_input = ts_input.ravel()
            elif len(ts_input.shape) == 1:
                self.n = ts_input.shape[0]
                if self.batch_size != 1:
                    raise ValueError("HoltWinters initialized with " +
                                     str(self.batch_size) + " time "
                                     "series, but data has size 1.")
            else:
                raise ValueError("Data must have 1 or 2 dimensions.")

        if self.n < 2*self.frequency:
            raise ValueError("Length of time series (" + str(self.n) +
                             ") must be at least double the frequency.")

        X_m, input_ptr, _, _, self.dtype = \
            input_to_dev_array(ts_input, order='C')

        cdef double[::1] alpha_d, beta_d, gamma_d, SSE_error_d, forecast_d
        cdef float[::1] alpha_f, beta_f, gamma_f, SSE_error_f, forecast_f

        if self.dtype == np.float32:
            alpha_f = np.ascontiguousarray(np.empty(self.batch_size,
                                                    dtype=self.dtype))
            beta_f = np.ascontiguousarray(np.empty(self.batch_size,
                                                   dtype=self.dtype))
            gamma_f = np.ascontiguousarray(np.empty(self.batch_size,
                                                    dtype=self.dtype))
            SSE_error_f = np.ascontiguousarray(np.empty(self.batch_size,
                                                        dtype=self.dtype))
            forecast_f = np.ascontiguousarray(np.empty(self.batch_size*self.h,
                                                       dtype=self.dtype))

            HoltWintersFitPredict(<int> self.n, <int> self.batch_size,
                                  <int> self.frequency, <int> self.h,
                                  <int> self.start_periods,
                                  <SeasonalType> self._cpp_stype,
                                  <float*> input_ptr,
                                  <float*> &alpha_f[0],
                                  <float*> &beta_f[0],
                                  <float*> &gamma_f[0],
                                  <float*> &SSE_error_f[0],
                                  <float*> &forecast_f[0])

            self.alpha = alpha_f
            self.beta = beta_f
            self.gamma = gamma_f
            self.SSE_error = SSE_error_f
            self.forecasted_points = forecast_f

        elif self.dtype == np.float64:
            alpha_d = np.ascontiguousarray(np.empty(self.batch_size,
                                                    dtype=self.dtype))
            beta_d = np.ascontiguousarray(np.empty(self.batch_size,
                                                   dtype=self.dtype))
            gamma_d = np.ascontiguousarray(np.empty(self.batch_size,
                                                    dtype=self.dtype))
            SSE_error_d = np.ascontiguousarray(np.empty(self.batch_size,
                                                        dtype=self.dtype))
            forecast_d = np.ascontiguousarray(np.empty(self.batch_size*self.h,
                                                       dtype=self.dtype))

            HoltWintersFitPredict(<int> self.n, <int> self.batch_size,
                                  <int> self.frequency, <int> self.h,
                                  <int > self.start_periods,
                                  <SeasonalType> self._cpp_stype,
                                  <double*> input_ptr,
                                  <double*> &alpha_d[0],
                                  <double*> &beta_d[0],
                                  <double*> &gamma_d[0],
                                  <double*> &SSE_error_d[0],
                                  <double*> &forecast_d[0])

            self.alpha = alpha_d
            self.beta = beta_d
            self.gamma = gamma_d
            self.SSE_error = SSE_error_d
            self.forecasted_points = forecast_d

        else:
            raise TypeError("HoltWinters supports only float32"
                            " and float64 input, but input type "
                            + str(self.dtype) + " passed.")

        self.fit_executed_flag = True
        del(X_m)

    def score(self, index):
        if index < 0 or index >= self.batch_size:
            raise IndexError("Index input: " + str(index) + " outside of "
                             "range [0, " + str(self.batch_size) + "]")
        if self.fit_executed_flag:
            return self.SSE_error[index]
        else:
            raise ValueError("Fit() the model before score()")

    def predict(self, index, h):
        if h > self.h:
            raise ValueError("Number of points must be <= pointsToForecast"
                             " (default = 50)."
                             " To get more points, execute fit() function"
                             " with pointsToForecast > 50. \nUsage : fit"
                             "(inputList, pointsToForecast) \n ")

        if self.fit_executed_flag:
            forecast = []
            if index < 0 or index >= self.batch_size:
                raise IndexError("Index input: " + str(index) + " outside of"
                                 " range [0, " + str(self.batch_size) + "]")

            # Get h points for nth time series forecast
            # from output 1d row major list
            for x in range(0, h):
                forecast.append(self.forecasted_points[self.h*index + x])
            return forecast
        else:
            raise ValueError("Fit() the model before predict()")

    def get_alpha(self, index):
        if index < 0 or index >= self.batch_size:
            raise IndexError("Index input: " + str(index) + " outside of "
                             "range [0, " + str(self.batch_size) + "]")
        if self.fit_executed_flag:
            return self.alpha[index]
        else:
            raise ValueError("Fit() the model to get alpha value")

    def get_beta(self, index):
        if index < 0 or index >= self.batch_size:
            raise IndexError("Index input: " + str(index) + " outside of "
                             "range [0, " + str(self.batch_size) + "]")
        if self.fit_executed_flag:
            return self.beta[index]
        else:
            raise ValueError("Fit() the model to get beta value")

    def get_gamma(self, index):
        if index < 0 or index >= self.batch_size:
            raise IndexError("Index input: " + str(index) + " outside of "
                             "range [0, " + str(self.batch_size) + "]")
        if self.fit_executed_flag:
            return self.gamma[index]
        else:
            raise ValueError("Fit() the model to get gamma value")
