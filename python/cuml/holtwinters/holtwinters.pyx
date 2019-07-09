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

import ctypes
import cudf
import numpy as np

from numba import cuda

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle

cdef extern from "holtwinters/Aion.hpp" namespace "aion":
    enum SeasonalType:
        ADDITIVE
        MULTIPLICATIVE
    
    cdef void HoltWintersFitPredict[Dtype](int n, int batch_size, int frequency, int h,
                                    int start_periods, SeasonalType seasonal,
                                    Dtype *data, Dtype *alpha_ptr, Dtype *beta_ptr,
                                    Dtype *gamma_ptr, Dtype *SSE_error_ptr, Dtype *forecast_ptr) except +
 

class Holtwinters(Base):
    
    def __init__(self,batch_size,freq_season,season_type,start_periods=2):

        self.batch_size = batch_size # Total number of Time Series for forecasting
        self.frequency =freq_season  # Season length in the time series
        self.season_type =season_type # Whether to perform additive or multiplicative STL decomposition
        self.forecasted_points = []  # list for final forecast output
        self.alpha  = [] # list for alpha values for each time series in batch
        self.beta   = []  # list for beta values for each time series in batch
        self.gamma  = []  # list for gamma values for each time series in batch
        self.SSE_error  = [] # SSE Error for all time series in batch 
        self.h = 50  # Default number of points to furecast in future
        self.fit_executed_flag = False 
        if freq_season < start_periods:
            raise Exception("Frequency cannot be less than 2 as number of seasons to be used for seasonal seed values is 2. \n ")
        else:	
            self.start_periods = start_periods # number of seasons to be used for seasonal seed values

    def fit(self, ts_input, pointsToForecast = 50):
        pass
   
    def score(self,index):

        index = index - 1
        if self.fit_executed_flag == True:
            return self.SSE_error[index]
        else:
            raise Exception("Fit() the model before score()")

    def predict(self,n,h):
        if h>50: 
            raise Exception("Default value of forecasted points is 50. To get more points, execute fit() function with pointsToForecast > 50. \nUsage : fit(numpyInputList, pointsToForecast) \n ")
            return	

        if self.fit_executed_flag == True:
            forecast =[]
            n = n-1

            # Get h points for nth time series forecast from output 1d row major list 
            for x in range(0,h):
                forecast.append(self.forecasted_points[self.h*n+x])
            return forecast
        else:
            raise Exception("Fit() the model before predict()")

    def get_alpha(self,index):
        	
        index = index - 1
        if self.fit_executed_flag == True:
            return self.alpha[index]
        else:
            raise Exception("Fit() the model to get alpha value")
   
    def get_beta(self,index):

        index = index - 1
        if self.fit_executed_flag == True:
            return self.beta[index]
        else:
            raise Exception("Fit() the model to get beta value")

    def get_gamma(self,index):

        index = index - 1
        if self.fit_executed_flag == True:
            return self.gamma[index]
        else:
            raise Exception("Fit() the model to get gamma value")



