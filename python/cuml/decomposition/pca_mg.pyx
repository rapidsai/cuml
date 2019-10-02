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
# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import ctypes
import cudf
import numpy as np

import rmm

from libcpp cimport bool
from libc.stdint cimport uintptr_t, uint32_t, uint64_t

from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.decomposition.utils cimport *
from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_dev_array, zeros

from cuml.decomposition import PCA

cdef extern from "cumlprims/opg/matrix/data.hpp" namespace "MLCommon::Matrix":

    cdef cppclass Data[T]:
        T *ptr
        size_t size

cdef extern from "cumlprims/opg/matrix/part_descriptor.hpp" namespace "MLCommon::Matrix":

    cdef cppclass RankSizePair:
        int rank
        size_t size

cdef extern from "cumlprims/opg/pca.hpp" namespace "ML::PCA::opg":

    cdef void fit(cumlHandle& handle,
                  RankSizePair **input,
                  Data[float] **rank_sizes,
                  float *components,
                  float *explained_var,
                  float *explained_var_ratio,
                  float *singular_vals,
                  float *mu,
                  float *noise_vars,
                  paramsPCA prms) except +

    cdef void fit(cumlHandle& handle,
                  RankSizePair **input,
                  Data[double] **rank_sizes,
                  double *input,
                  double *components,
                  double *explained_var,
                  double *explained_var_ratio,
                  double *singular_vals,
                  double *mu,
                  double *noise_vars,
                  paramsPCA prms) except +


class PCAMG(PCA):

    def __init__(self, **kwargs):
        super(PCAMG, self).__init__(**kwargs)

    def fit(self, X, M, N, partsToRanks):
        """
        Fit function for PCA MG. This not meant to be used as
        part of the public API.
        :param X: array of __cuda_array_interface__ for local parts
        :param M: total number of rows
        :param N: total number of cols
        :param partsToRanks: array of tuples in the format: [(rank,size)]
        :return: self
        """
        pass
