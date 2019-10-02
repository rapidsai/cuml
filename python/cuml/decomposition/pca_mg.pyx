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

from libc.stdlib cimport malloc, free


from libcpp cimport bool
from libc.stdint cimport uintptr_t, uint32_t, uint64_t

from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.decomposition.utils cimport *
from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_dev_array, zeros

from cuml.decomposition import PCA

cdef extern from "cumlprims/opg/matrix/data.hpp" namespace "MLCommon::Matrix":

    cdef cppclass floatData_t:
        float *ptr
        size_t totalSize

    cdef cppclass doubleData_t:
        double *ptr
        size_t totalSize

cdef extern from "cumlprims/opg/matrix/part_descriptor.hpp" namespace "MLCommon::Matrix":

    cdef cppclass RankSizePair:
        int rank
        size_t size

cdef extern from "cumlprims/opg/pca.hpp" namespace "ML::PCA::opg":

    cdef void fitF(cumlHandle& handle,
                  RankSizePair **input,
                  size_t n_parts,
                  floatData_t **rank_sizes,
                  float *components,
                  float *explained_var,
                  float *explained_var_ratio,
                  float *singular_vals,
                  float *mu,
                  float *noise_vars,
                  paramsPCA prms) except +

    cdef void fitD(cumlHandle& handle,
                  RankSizePair **input,
                  size_t n_parts,
                  doubleData_t **rank_sizes,
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

    def fit(self, X, M, N, partsToRanks, _transform=False):
        """
        Fit function for PCA MG. This not meant to be used as
        part of the public API.
        :param X: array of local dataframes / array partitions
        :param M: total number of rows
        :param N: total number of cols
        :param partsToRanks: array of tuples in the format: [(rank,size)]
        :return: self
        """

        print("M=" + str(M))
        print("N=" + str(N))

        # TODO: Create outputs, convert X to **Data, use M, N to build paramsPCA, & partsToRanks to build **RankSizePair
        arr_interfaces = []
        cdef uintptr_t input_ptr
        for arr in X:
            X_m, input_ptr, n_rows, n_cols, dtype = \
                input_to_dev_array(arr, check_dtype=[np.float32, np.float64])
            arr_interfaces.append({"obj": X_m, "data": input_ptr, "shape": (n_rows, n_cols)})

        n_parts = len(X)
        cdef floatData_t **dataF = <floatData_t**> malloc(sizeof(floatData_t*) * len(X))
        cdef doubleData_t **dataD = <doubleData_t**> malloc(sizeof(doubleData_t*) * len(X))
        for x_i in range(len(arr_interfaces)):
            x = arr_interfaces[x_i]
            input_ptr = x["data"]
            dataF[x_i] = <floatData_t*>malloc(sizeof(floatData_t))
            dataF[x_i].ptr = <float*>input_ptr
            dataF[x_i].totalSize = <size_t>x["shape"][0]

        cdef RankSizePair **rankSizePair = <RankSizePair**>malloc(sizeof(RankSizePair**)
                                                                  * len(partsToRanks))
        for idx, rankSize in enumerate(partsToRanks):
            rank, size = rankSize
            rankSizePair[idx] = <RankSizePair*> malloc(sizeof(RankSizePair))
            rankSizePair[idx].rank = <int>rank
            rankSizePair[idx].size = <size_t>size

        cpdef paramsPCA params
        params.n_components = self.n_components
        params.n_rows = int(M)
        params.n_cols = int(N)
        params.whiten = self.whiten
        params.n_iterations = self.iterated_power
        params.tol = self.tol
        params.algorithm = self.c_algorithm

        if self.n_components > self.n_cols:
            raise ValueError('Number of components should not be greater than'
                             'the number of columns in the data')

        self._initialize_arrays(params.n_components,
                                params.n_rows, params.n_cols)

        cdef uintptr_t comp_ptr = get_dev_array_ptr(self.components_ary)

        cdef uintptr_t explained_var_ptr = \
            get_cudf_column_ptr(self.explained_variance_)

        cdef uintptr_t explained_var_ratio_ptr = \
            get_cudf_column_ptr(self.explained_variance_ratio_)

        cdef uintptr_t singular_vals_ptr = \
            get_cudf_column_ptr(self.singular_values_)

        cdef uintptr_t mean_ptr = get_cudf_column_ptr(self.mean_)

        cdef uintptr_t noise_vars_ptr = \
            get_cudf_column_ptr(self.noise_variance_)

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        if self.dtype == np.float32:
            fitF(handle_[0],
                <RankSizePair**>rankSizePair,
                <size_t> n_parts,
                <floatData_t**>dataF,
                <float*> comp_ptr,
                <float*> explained_var_ptr,
                <float*> explained_var_ratio_ptr,
                <float*> singular_vals_ptr,
                <float*> mean_ptr,
                <float*> noise_vars_ptr,
                params)
        else:
            fitD(handle_[0],
                <RankSizePair**>rankSizePair,
                <size_t> n_parts,
                <doubleData_t**>dataF,
                <double*> comp_ptr,
                <double*> explained_var_ptr,
                <double*> explained_var_ratio_ptr,
                <double*> singular_vals_ptr,
                <double*> mean_ptr,
                <double*> noise_vars_ptr,
                params)

        # make sure the previously scheduled gpu tasks are complete before the
        # following transfers start
        self.handle.sync()

        # Keeping the additional dataframe components during cuml 0.8.
        # See github issue #749
        self.components_ = cudf.DataFrame()
        for i in range(0, params.n_cols):
            n_c = params.n_components
            self.components_[str(i)] = self.components_ary[i*n_c:(i+1)*n_c]

        if (isinstance(X, cudf.DataFrame)):
            del(X_m)

        if not _transform:
            del(self.trans_input_)

        return self


