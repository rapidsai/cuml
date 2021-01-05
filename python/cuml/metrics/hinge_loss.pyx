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

from libc.stdint cimport uintptr_t

from cuml.common import input_to_cuml_array
from cuml.metrics.pairwise_distances import _determine_metric
from cuml.raft.common.handle cimport handle_t
from cuml.raft.common.handle import Handle
from cuml.metrics.distance_type cimport DistanceType
from cuml.metrics.penalty_type cimport penalty as PenaltyType


cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics":
    double hinge_loss(const handle_t &handle,
                    double *input,
                    int n_rows,
                    int n_cols,
                    const double *labels,
                    const double *coef,
                    PenaltyType pen,
                    double alpha,
                    double l1_ratio) except +

def _determine_penalty(penalty_str):

    if penalty_str == 'none':
        return PenaltyType.PenaltyNone
    elif penalty_str == 'L1':
        return PenaltyType.PenaltyL1
    elif penalty_str == 'L2':
        return PenaltyType.PenaltyL2
    elif penalty_str == 'elasticnet':
        return PenaltyType.PenaltyElasticNet
    else:
        raise ValueError(" The metric: '{}', is not supported at this time."
                         .format(penalty_str))
   
        
def cython_hinge_loss(
        X, labels, coef, penalty_str, alpha, l1_ratio, handle=None):
    """
    """
    handle = Handle() if handle is None else handle
    cdef handle_t *handle_ = <handle_t*> <size_t> handle.getHandle()

    data, n_rows, n_cols, _ = input_to_cuml_array(
        X,
        order='C',
        convert_to_dtype=np.float64
    )

    labels, _, _, _ = input_to_cuml_array(
        labels,
        order='C',
        convert_to_dtype=np.float64
    )
    
    coef, _, _, _ = input_to_cuml_array(
        labels,
        order='C',
        convert_to_dtype=np.float64
    )

    penalty = _determine_penalty(penalty_str)
    return hinge_loss(handle_[0],
                      <double*> <uintptr_t> data.ptr,
                      n_rows,
                      n_cols,
                      <double*> <uintptr_t> labels.ptr,
                      <double*> <uintptr_t> coef.ptr,
                      penalty,
                      alpha,
                      l1_ratio)