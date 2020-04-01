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

import cupy as cp

import cuml
from cuml.common.array import CumlArray as cumlArray
from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.tsa.arima import ARIMA
from cuml.tsa.seasonality import seas_test
from cuml.tsa.stationarity import kpss_test
from cuml.tsa.utils import divide_by_mask, divide_by_min
from cuml.utils.input_utils import input_to_cuml_array

tests_map = {
    "kpss": kpss_test,
    "seas": seas_test,
}

# TODO:
# - stepwise argument? -> if false, complete traversal
# - approximation argument to choose the model based on conditional sum of
#   squares?
# - Box-Cox transformations? (parameter lambda)

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
            seasonal_test="seas",
            verbose=False):
        """TODO: docs
        """
        # Notes:
        #  - We iteratively divide the dataset as we decide parameters, so
        #    it's important to make sure that we don't keep the unused arrays
        #    alive, so they can get garbage collected.
        #  - As we divide the dataset, we also keep track of the original
        #    index of each series in the batch, to construct the final map at
        #    the end.

        # Parse input parameters
        ic = ic.lower()
        test = test.lower()
        seasonal_test = seasonal_test.lower()
        if s == 1:  # R users might use s=1 for a non-seasonal dataset
            s = None

        # Box-Cox transform
        # TODO: handle it

        # Original index
        d_index, *_ = input_to_cuml_array(np.r_[:self.batch_size],
                                          convert_to_dtype=np.int32)
        # TODO: worth building on GPU?

        #
        # Choose the hyper-parameter D
        #
        if verbose:
            print("Deciding D...")
        if not s:
            # Non-seasonal -> D=0
            data_D = {0: (self.d_y, d_index)}
        elif D is not None:
            # D is specified by the user
            data_D = {D: (self.d_y, d_index)}
        else:
            # D is chosen with a seasonal differencing test
            if seasonal_test not in tests_map:
                raise ValueError("Unknown seasonal diff test: {}"
                                 .format(seasonal_test))
            mask = tests_map[seasonal_test](self.d_y, s)
            data_D = {}
            (out0, index0), (out1, index1) = divide_by_mask(self.d_y, mask,
                                                            d_index)
            if out0 is not None:
                data_D[0] = (out0, index0)
            if out1 is not None:
                data_D[1] = (out1, index1)
            del mask, out0, index0, out1, index1
        # TODO: can D be 2?

        #
        # Choose the hyper-parameter d
        #
        if verbose:
            print("Deciding d...")
        data_dD = {}
        for D_ in data_D:
            if d is not None:
                # d is specified by the user
                data_dD[(d, D_)] = data_D[D_]
            else:
                # d is decided with stationarity tests
                if test not in tests_map:
                    raise ValueError("Unknown stationarity test: {}"
                                     .format(test))
                data_temp, id_temp = data_D[D_]
                for d_ in range(min(max_d, 2 - D_)):
                    mask = tests_map[test](data_temp, d_, D_, s)
                    (out0, index0), (out1, index1) \
                        = divide_by_mask(data_temp, mask, id_temp)
                    if out1 is not None:
                        data_dD[(d_, D_)] = (out1, index1)
                    if out0 is not None:
                        (data_temp, id_temp) = (out0, index0)
                    else:
                        break
                else: # (when the for loop reaches its end naturally)
                    # The remaining series are assigned the max possible d
                    data_dD[(min(max_d, 2 - D_), D_)] = (data_temp, id_temp)
                del data_temp, id_temp, mask, out0, index0, out1, index1
        del data_D

        # Limit the number of parameters to what we can handle
        # TODO: handle more than 4 in the Jones transform
        max_p = min(max_p, 4)
        max_q = min(max_q, 4)
        if s:
            max_p = min(max_p, s - 1)
            max_q = min(max_q, s - 1)
        max_P = min(max_P, 4) if s else 0
        max_Q = min(max_Q, 4) if s else 0
        start_p = min(start_p, max_p)
        start_q = min(start_q, max_p)
        start_P = min(start_P, max_p)
        start_Q = min(start_Q, max_p)

        #
        # Choose the hyper-parameters p, q, P, Q, k
        #
        data_complete = {}
        for (d_, D_) in data_dD:
            data_temp, id_temp = data_dD[(d_, D_)]
            batch_size = data_temp.shape[1] if len(data_temp.shape) > 1 else 1
            k_ = 1 if d_ + D_ <= 1 else 0

            # Grid search
            # TODO: think about a (partially) step-wise parallel approach
            all_ic = []
            all_orders = []
            for p_ in range(start_p, max_p + 1):
                for q_ in range(start_q, max_q + 1):
                    for P_ in range(start_P, max_P + 1):
                        for Q_ in range(start_Q, max_Q + 1):
                            s_ = s if (P_ + D_ + Q_) else 0
                            # TODO: raise issue that input_to_cuml_array
                            #       should support cuML arrays
                            model = ARIMA(cp.asarray(data_temp),
                                          order=(p_, d_, q_),
                                          seasonal_order=(P_, D_, Q_, s_),
                                          fit_intercept=k_,
                                          handle=self.handle)
                            if verbose:
                                print("Fitting {}".format(model))
                            model.fit()  # TODO: support approximation
                            all_ic.append(model._ic(ic))
                            all_orders.append((p_, q_, P_, Q_, s_, k_))

            # Organize the results into a matrix
            n_models = len(all_orders)
            ic_matrix, *_ = input_to_cuml_array(
                cp.concatenate([cp.asarray(ic_arr).reshape(batch_size, 1)
                                for ic_arr in all_ic], 1))

            # Divide the batch, choosing the best model for each series
            sub_batches, sub_id = divide_by_min(data_temp, ic_matrix, id_temp)
            for i in range(n_models):
                if sub_batches[i] is None:
                    continue
                p_, q_, P_, Q_, s_, k_ = all_orders[i]
                data_complete[(p_, d_, q_, P_, D_, Q_, s_, k_)] \
                    = (sub_batches[i], sub_id[i])

            del model, all_ic, all_orders, ic_matrix, sub_batches, sub_id

        # TODO: try different k_ on the best model?

        # TODO: build map
        self.id_to_pos = None
        self.id_to_model = None

        return data_complete
