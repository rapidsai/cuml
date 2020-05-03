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

import numpy as np

from cuml.utils.opg_data_utils cimport *
from libc.stdlib cimport malloc, free
from libc.stdint cimport uintptr_t, uint32_t, uint64_t


def build_data_t(arys):
    cdef floatData_t **data_f32
    cdef doubleData_t **data_f64
    cdef uintptr_t ary_ptr

    if arys[0].dtype == np.float32:
        data_f32 = <floatData_t **> malloc(
            sizeof(floatData_t *) * len(arys))

        for idx in range(len(arys)):
            data_f32[idx] = <floatData_t*> malloc(sizeof(floatData_t))
            ary_ptr = arys[idx].ptr
            data_f32[idx].ptr = <float*> ary_ptr
            data_f32[idx].totalSize = len(arys[idx])

        return <size_t>data_f32

    elif arys[0].dtype == np.float64:
        data_f64 = <doubleData_t **> malloc(
            sizeof(doubleData_t *) * len(arys))

        for idx in range(len(arys)):
            data_f64[idx] = <doubleData_t*> malloc(sizeof(doubleData_t))
            ary_ptr = arys[idx].ptr
            data_f64[idx].ptr = <double*> ary_ptr
            data_f64[idx].totalSize = len(arys[idx])

        return <size_t>data_f64

    else:
        raise TypeError('build_data_t: Arrays passed must be np.float32 or \
                        np.float64')


def free_data_t(data_t, n, dtype):
    cdef uintptr_t data_ptr = data_t
    cdef floatData_t **d32
    cdef doubleData_t **d64

    if dtype == np.float32:
        d32 = <floatData_t**>data_ptr
        for x_i in range(n):
            free(d32[x_i])
        free(d32)
    else:
        d64 = <doubleData_t**>data_ptr
        for x_i in range(n):
            free(d64[x_i])
        free(d64)


def build_rank_size_pair(arys, rank):
    cdef RankSizePair **rankSizePair = <RankSizePair**> \
        malloc(sizeof(RankSizePair**)
               * len(arys))

    for i in range(len(arys)):
        rankSizePair[i] = <RankSizePair*> \
            malloc(sizeof(RankSizePair))
        rankSizePair[i].rank = <int>rank
        rankSizePair[i].size = <size_t>len(arys[i][0])

    return <size_t> rankSizePair


def free_rank_size_pair(rank_size_t, n):
    cdef uintptr_t rs_ptr = rank_size_t
    cdef RankSizePair **rankSizePair = <RankSizePair**> rs_ptr

    for idx in range(n):
        free(<RankSizePair*>rankSizePair[idx])
    free(<RankSizePair**>rankSizePair)


def _build_dataFloat(arr_interfaces):
    cdef floatData_t **dataF = <floatData_t **> \
        malloc(sizeof(floatData_t *)
               * len(arr_interfaces))

    cdef uintptr_t input_ptr
    for x_i in range(len(arr_interfaces)):
        x = arr_interfaces[x_i]
        input_ptr = x["data"]
        dataF[x_i] = <floatData_t *> malloc(sizeof(floatData_t))
        dataF[x_i].ptr = <float *> input_ptr
        dataF[x_i].totalSize = <size_t> x["shape"][0]
    return <size_t>dataF


def _build_dataDouble(arr_interfaces):
    cdef doubleData_t **dataD = <doubleData_t **> \
        malloc(sizeof(doubleData_t *)
               * len(arr_interfaces))

    cdef uintptr_t input_ptr
    for x_i in range(len(arr_interfaces)):
        x = arr_interfaces[x_i]
        input_ptr = x["data"]
        dataD[x_i] = <doubleData_t *> malloc(sizeof(doubleData_t))
        dataD[x_i].ptr = <double *> input_ptr
        dataD[x_i].totalSize = <size_t> x["shape"][0]
    return <size_t>dataD


def _freeDoubleD(data, arr_interfaces):
    cdef uintptr_t data_ptr = data
    cdef doubleData_t **d = <doubleData_t**>data_ptr
    for x_i in range(len(arr_interfaces)):
        free(d[x_i])
    free(d)


def _freeFloatD(data, arr_interfaces):
    cdef uintptr_t data_ptr = data
    cdef floatData_t **d = <floatData_t**>data_ptr
    for x_i in range(len(arr_interfaces)):
        free(d[x_i])
    free(d)


def _build_predData(partsToSizes, rank, n_cols, dtype):
    arr_interfaces_trans = []
    for idx, rankSize in enumerate(partsToSizes):
        rk, size = rankSize
        if rank == rk:
            trans_ary = CumlArray.zeros((size, n_cols),
                                        order="F",
                                        dtype=dtype)

            arr_interfaces_trans.append({"obj": trans_ary,
                                         "data": trans_ary.ptr,
                                         "shape": (size, n_cols)})

    return arr_interfaces_trans
