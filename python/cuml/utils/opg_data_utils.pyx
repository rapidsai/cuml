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
    """
    Function to create a floatData_t** or doubleData_t** from a list of
    cumlArrays

    Parameters
    ----------

    arys: list of cumlArrays of the same dtype, np.float32 or np.float64

    Returns
    -------
    ptr: pointer to either a floatData_t** or doubleData_t**, depending on
        dtype of input

    """
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
    """
    Function to free a floatData_t** or doubleData_t**

    Parameters
    ----------
    data_t: a floatData_t** or doubleData_t**
    n: number of elements in the floatData_t** or doubleData_t**
    dtype: np.float32 or np.float64 indicating whether data_t is a
        floatData_t** or doubleData_t**
    """

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
    """
    Funciton to build a rankSizePair** mapping the rank to the sizes of arrays
    in arys

    Parameters
    ----------
    arys: list of arrays (usual acceptable array/dataframe formats)
    rank: rank to be mapped

    Returns:
    --------
    ptr: pointer to the rankSizePair**
    """
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
    """
    Function to free a rankSizePair**

    Parameters
    ----------
    rank_size_t: rankSizePair** to be freed.
    n: number of elements in the rankSizePair** to be freed.
    """
    cdef uintptr_t rs_ptr = rank_size_t
    cdef RankSizePair **rankSizePair = <RankSizePair**> rs_ptr

    for idx in range(n):
        free(<RankSizePair*>rankSizePair[idx])
    free(<RankSizePair**>rankSizePair)
