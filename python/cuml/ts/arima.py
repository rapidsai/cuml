#
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

from typing import List, Tuple
import numpy as np
from IPython.core.debugger import set_trace

# from cuml.ts.batched_kalman import batched_kfilter
from cuml.ts.batched_arima import pynvtx_range_push, pynvtx_range_pop
from cuml.ts.batched_arima import pack, unpack

import cuml.ts.batched_arima as batched_arima

import cudf



def diffAndCenter(y: np.ndarray,
                  p, q,
                  mu_ar_ma_params_x: np.ndarray):
    """Diff and center batched series `y`"""
    pynvtx_range_push("diffAndCenter")
    y_diff = np.diff(y, axis=0)

    pynvtx_range_pop()
    return np.asfortranarray(y_diff-mu_ar_ma_params_x[::(1+p+q)])


# def run_kalman(y, order: Tuple[int, int, int],
#                num_batches, mu_ar_ma_params_x,
#                initP_kalman_iterations=False) -> Tuple[np.ndarray, np.ndarray]:
#     """Run the (batched) kalman filter for the given model (and contained batched
#     series). `initP_kalman_iterations, if true uses kalman iterations, and if false
#     uses an analytical approximation (Durbin Koopman pg 138).`"""
#     pynvtx_range_push("run_kalman")
#     p, d, q = order

#     if d == 0:

#         ll_b, vs = batched_kfilter(np.asfortranarray(y), # numpy
#                                    mu_ar_ma_params_x,
#                                    p, d, q,
#                                    initP_kalman_iterations)
#     elif d == 1:

#         y_diff_centered = diffAndCenter(y, p, q, mu_ar_ma_params_x)
#         # print("ydiff:", y_diff_centered)
#         ll_b, vs = batched_kfilter(y_diff_centered, # numpy
#                                    mu_ar_ma_params_x,
#                                    p, d, q,
#                                    initP_kalman_iterations)
#     else:
#         raise NotImplementedError("ARIMA only support d==0,1")

#     pynvtx_range_pop()
#     return ll_b, vs


# def predict_in_sample(model):
#     """Return in-sample prediction on batched series given batched model"""

#     p, d, q = model.order[0]
#     x = pack(p, d, q, model.num_batches, model.mu, model.ar_params, model.ma_params)
#     vs = batched_arima.residual(model.num_batches, model.num_samples, model.order[0], model.y, x)
#     # _, vs = run_kalman(model.y, (p, d, q), model.num_batches, x)

#     assert_same_d(model.order) # We currently assume the same d for all series
#     _, d, _ = model.order[0]

#     if d == 0:
#         y_p = model.y - vs
#     elif d == 1:
#         y_diff = np.diff(model.y, axis=0)
#         # Following statsmodel `predict(typ='levels')`, by adding original
#         # signal back to differenced prediction, we retrive a prediction of
#         # the original signal.
#         predict = (y_diff - vs)
#         y_p = model.y[0:-1, :] + predict
#     else:
#         # d>1
#         raise NotImplementedError("Only support d==0,1")

#     # Extend prediction by 1 when d==1
#     if d == 1:
#         # forecast a single value to make prediction length of original signal
#         fc1 = np.zeros(model.num_batches)
#         for i in range(model.num_batches):
#             fc1[i] = fc_single(1, model.order[i], y_diff[:,i],
#                                vs[:,i], model.mu[i],
#                                model.ma_params[i],
#                                model.ar_params[i])

#         final_term = model.y[-1, :] + fc1

#         # append final term to prediction
#         temp = np.zeros((y_p.shape[0]+1, y_p.shape[1]))
#         temp[:-1, :] = y_p
#         temp[-1, :] = final_term
#         y_p = temp

#     model.yp = y_p
#     return y_p







