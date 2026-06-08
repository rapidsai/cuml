#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t
from libc.stdlib cimport free, malloc

from cuml.common.opg_data_utils_mg cimport *


def build_data_t(parts):
    """
    Function to create a floatData_t** or doubleData_t** from a list of
    cupy arrays

    Parameters
    ----------
    parts: list[cp.ndarray]
        A list of cupy arrays, all with the same dtype (float32 or float64).

    Returns
    -------
    ptr: vector pointer of either a floatData_t* or doubleData_t*,
         depending on dtype of input

    """
    cdef vector[floatData_t *] *data_f32
    cdef vector[doubleData_t *] *data_f64

    cdef floatData_t *data_f
    cdef doubleData_t *data_d

    if parts[0].dtype == np.float32:
        data_f32 = new vector[floatData_t *]()
        for part in parts:
            data_f = <floatData_t*> malloc(sizeof(floatData_t))
            data_f.ptr = <float*><uintptr_t>(part.data.ptr)
            data_f.totalSize = len(part)
            data_f32.push_back(data_f)

        return <uintptr_t> data_f32

    elif parts[0].dtype == np.float64:
        data_f64 = new vector[doubleData_t *]()
        for part in parts:
            data_d = <doubleData_t*> malloc(sizeof(doubleData_t))
            data_d.ptr = <double*><uintptr_t>(part.data.ptr)
            data_d.totalSize = len(part)
            data_f64.push_back(data_d)

        return <uintptr_t> data_f64

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
        del d32
    else:
        d64 = <vector[doubleData_t*]*> data_ptr
        for x_i in range(d64.size()):
            free(d64.at(x_i))
        del d64


def build_rank_size_pair(parts_to_sizes, rank):
    """
    Function to build a vector<rankSizePair*> mapping the rank to the
    sizes of partitions

    Parameters
    ----------
    parts_to_sizes: array of tuples in the format: [(rank,size)]
    rank: rank to be mapped

    Returns
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
    del rsp_vec


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

    Returns
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

    return <uintptr_t>descriptor


def free_part_descriptor(descriptor_ptr):
    """
    Function to free a PartDescriptor*
    Parameters
    ----------
    descriptor_ptr: PartDescriptor* to be freed
    """
    cdef PartDescriptor *desc_c = <PartDescriptor*><size_t>descriptor_ptr
    del desc_c
