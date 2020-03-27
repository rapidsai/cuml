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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import ctypes
import numpy as np
from libc.stdint cimport uintptr_t
from libcpp cimport bool

import cuml
from cuml.common.array import CumlArray as cumlArray
from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.tsa.arima import ARIMA
from cuml.tsa.seasonality import seas_test
from cuml.tsa.stationarity import kpss_test
from cuml.tsa.utils import divide_batch
from cuml.utils.input_utils import input_to_cuml_array

tests_map = {
    "kpss": kpss_test,
    "seas": seas_test,
}

# TODO:
# - stepwise argument? -> if false, complete traversal
# - approximation argument to choose the model based on conditional sum of
#   squares?

class AutoARIMA(Base):
    r"""TODO: docs
    """
    
    def __init__(self, y, handle=None):
        super().__init__(handle)

        # Get device array. Float64 only for now.
        self.d_y, self.n_obs, self.batch_size, self.dtype \
            = input_to_cuml_array(y, check_dtype=np.float64)

    def fit(self,
            s=None,
            d=None,
            D=None,
            max_d=2,
            max_D=1,
            start_p=2, # TODO: start at 0?
            start_q=2,
            start_P=1,
            start_Q=1,
            max_p=4, # TODO: support p=5 / q=5 in ARIMA
            max_q=4,
            max_P=2,
            max_Q=2,
            ic="aicc", # TODO: which one to use by default?
            test="kpss",
            seasonal_test="seas"):
        """TODO: docs
        """
        ic = ic.lower()
        test = test.lower()
        seasonal_test = seasonal_test.lower()

        # Notes:
        #  - We iteratively divide the dataset as we decide parameters, so
        #    it's important to make sure that we don't keep the unused arrays
        #    alive, so they can get garbage collected.
        #  - As we divide the dataset, we also keep track of the original
        #    index of each series in the batch, to construct the final map at
        #    the end.

        # Original index
        d_index, *_ = input_to_cuml_array(np.r_[:self.batch_size],
                                          convert_to_dtype=np.int32)
        # TODO: worth building on GPU?

        # TODO: manage empty arrays during divisions!

        #
        # Decide between D=0 and D=1
        #
        if not s:
            data_D = {0: (self.d_y, d_index)}
        elif D is not None:
            data_D = {D: (self.d_y, d_index)}
        else:
            if seasonal_test not in tests_map:
                raise ValueError("Unknown seasonal diff test: {}".format(s))
            mask = tests_map[seasonal_test](self.d_y, s)
            data_D = {}
            (out0, index0), (out1, index1) = divide_batch(self.d_y, mask,
                                                          d_index)
            if out0 is not None:
                data_D[0] = (out0, index0)
            if out1 is not None:
                data_D[1] = (out1, index1)
            del mask, out0, index0, out1, index1
        # TODO: can D be 2?

        #
        # Decide the value of d
        #
        data_dD = {}
        for D_ in data_D:
            if d is not None:
                data_dD[(d, D_)] = data_D[D_]
            else:
                if test not in tests_map:
                    raise ValueError("Unknown stationarity test: {}"
                                     .format(s))
                data_temp, id_temp = data_D[D_]
                for d_ in range(min(max_d, 2 - D_)):
                    mask = tests_map[test](data_temp, d_, D_, s)
                    (out0, index0), (out1, index1) \
                        = divide_batch(data_temp, mask, id_temp)
                    if out1 is not None:
                        data_dD[(d_, D_)] = (out1, index1)
                    if out0 is not None:
                        (data_temp, id_temp) = (out0, index0)
                    else:
                        break
                else: # (in case the for loop reaches its end naturally)
                    # Remaining series are assigned the max possible d
                    data_dD[(min(max_d, 2 - D_), D_)] = (data_temp, id_temp)
                del data_temp, id_temp, mask, out0, index0, out1, index1
        del data_D

        # Temporary for debug
        return data_dD
