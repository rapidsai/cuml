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

from cuml.common.opg_data_utils_mg cimport *
from libc.stdlib cimport malloc, free
from libc.stdint cimport uintptr_t, uint32_t, uint64_t
from cython.operator cimport dereference as deref
from cuml.common.array import CumlArray


def build_data_t(arys):
    """
    Function to create a floatData_t** or doubleData_t** from a list of
    cumlArrays

    Parameters
    ----------

    arys: list of cumlArrays of the same dtype, np.float32 or np.float64

    Returns
    -------
    ptr: vector pointer of either a floatData_t* or doubleData_t*,
         depending on dtype of input

    """
    cdef vector[floatData_t *] *data_f32 = new vector[floatData_t *]()
    cdef vector[doubleData_t *] *data_f64 = new vector[doubleData_t *]()

    cdef uintptr_t ary_ptr
    cdef floatData_t *data_f
    cdef doubleData_t *data_d
    cdef uintptr_t data_ptr

    if arys[0].dtype == np.float32:

        for idx in range(len(arys)):
            data_f = <floatData_t*> malloc(sizeof(floatData_t))
            ary_ptr = arys[idx].ptr
            data_f.ptr = <float*> ary_ptr
            data_f.totalSize = len(arys[idx])
            data_f32.push_back(data_f)

        data_ptr = <uintptr_t> data_f32
        return data_ptr

    elif arys[0].dtype == np.float64:

        for idx in range(len(arys)):
            data_d = <doubleData_t*> malloc(sizeof(doubleData_t))
            ary_ptr = arys[idx].ptr
            data_d.ptr = <double*> ary_ptr
            data_d.totalSize = len(arys[idx])
            data_f64.push_back(data_d)

        data_ptr = <uintptr_t> data_f64
        return data_ptr

    else:
        raise TypeError('build_data_t: Arrays passed must be np.float32 or \
                        np.float64')


def free_data_t(data_t, dtype):
    """
    Function to free a vector of floatData_t* or doubleData_t*

    Parameters
    ----------
    data_t: a vector of floatData_t* or doubleData_t*
    dtype: np.float32 or np.float64 indicating whether data_t is a
        floatData_t* or doubleData_t*
    """
    cdef uintptr_t data_ptr = data_t 

    cdef vector[floatData_t*] *d32
    cdef vector[doubleData_t*] *d64

    if dtype == np.float32:
        d32 = <vector[floatData_t*]*> data_ptr
        for x_i in range(d32.size()):
            free(d32.at(x_i))
        free(d32)
    else:
        d64 = <vector[doubleData_t*]*> data_ptr
        for x_i in range(d64.size()):
            free(d64.at(x_i))
        free(d64)


def build_rank_size_pair(parts_to_sizes, rank):
    """
    Function to build a vector<rankSizePair*> mapping the rank to the
    sizes of partitions

    Parameters
    ----------
    parts_to_sizes: array of tuples in the format: [(rank,size)]
    rank: rank to be mapped

    Returns:
    --------
    ptr: vector pointer of the RankSizePair*
    """
    cdef vector[RankSizePair*] *rsp_vec = new vector[RankSizePair*]()

    for idx, rankToSize in enumerate(parts_to_sizes):
        rank, size = rankToSize
        rsp = <RankSizePair*> malloc(sizeof(RankSizePair))
        rsp.rank = <int>rank
        rsp.size = <size_t>size

        rsp_vec.push_back(rsp)

    cdef uintptr_t rsp_ptr = <uintptr_t> rsp_vec
    return rsp_ptr


def free_rank_size_pair(rank_size_t):
    """
    Function to free a vector of rankSizePair*

    Parameters
    ----------
    rank_size_t: vector of rankSizePair* to be freed.
    """
    cdef uintptr_t rank_size_ptr = rank_size_t 

    cdef vector[RankSizePair *] *rsp_vec \
        = <vector[RankSizePair *]*> rank_size_ptr

    for x_i in range(rsp_vec.size()):
        free(rsp_vec.at(x_i))
    free(rsp_vec)


def build_part_descriptor(m, n, rank_size_t, rank):
    """
    Function to build a shared PartDescriptor object

    Parameters
    ----------
    m: total number of rows across all workers
    n: number of cols
    rank_size_t: vector of rankSizePair * to be used for
        building the part descriptor
    rank: rank to be mapped

    Returns:
    --------
    ptr: PartDescriptor object
    """
    cdef uintptr_t rank_size_ptr = rank_size_t 

    cdef vector[RankSizePair *] *rsp_vec \
        = <vector[RankSizePair *]*> rank_size_ptr

    cdef PartDescriptor *descriptor \
        = new PartDescriptor(<size_t>m,
                             <size_t>n,
                             <vector[RankSizePair*]>deref(rsp_vec),
                             <int>rank)

    cdef uintptr_t desc_ptr = <uintptr_t>descriptor
    return desc_ptr


def free_part_descriptor(descriptor_ptr):
    """
    Function to free a PartDescriptor*

    Parameters
    ----------
    descriptor_ptr: PartDescriptor* to be freed
    """
    cdef PartDescriptor *desc_c \
        = <PartDescriptor*><size_t>descriptor_ptr
    free(desc_c)


def build_pred_or_trans_arys(arys, order, dtype):
    output_arys = []
    for i in range(len(arys)):
        out = CumlArray.zeros(arys[i].shape,
                              order=order,
                              dtype=dtype)

        output_arys.append(out)

    return output_arys
