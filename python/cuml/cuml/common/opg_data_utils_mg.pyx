#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t

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

    if parts[0].dtype == np.float32:
        data_f32 = new vector[floatData_t *]()
        for part in parts:
            data_f32.push_back(
                new floatData_t(
                    <float*><uintptr_t>(part.data.ptr),
                    len(part),
                )
            )
        return <uintptr_t> data_f32

    elif parts[0].dtype == np.float64:
        data_f64 = new vector[doubleData_t *]()
        for part in parts:
            data_f64.push_back(
                new doubleData_t(
                    <double*><uintptr_t>(part.data.ptr),
                    len(part),
                )
            )
        return <uintptr_t> data_f64

    else:
        raise TypeError('build_data_t: Arrays passed must be np.float32 or \
                        np.float64')


def free_data_t(uintptr_t data_ptr, dtype):
    """
    Function to free a vector of floatData_t* or doubleData_t*

    Parameters
    ----------
    data_ptr: int
        A pointer to a vector of floatData_t* or doubleData_t*.
    dtype: dtype
        The dtype (float32 or float64).
    """
    cdef vector[floatData_t*] *d32
    cdef vector[doubleData_t*] *d64
    cdef floatData_t* ptr_32
    cdef doubleData_t* ptr_64

    if dtype == np.float32:
        d32 = <vector[floatData_t*]*> data_ptr
        for x_i in range(d32.size()):
            ptr_32 = d32.at(x_i)
            del ptr_32
        del d32
    else:
        d64 = <vector[doubleData_t*]*> data_ptr
        for x_i in range(d64.size()):
            ptr_64 = d64.at(x_i)
            del ptr_64
        del d64


def build_rank_size_pair(parts_to_sizes, rank):
    """
    Function to build a vector<rankSizePair*> mapping the rank to the
    sizes of partitions

    Parameters
    ----------
    parts_to_sizes: array of tuples in the format: [(rank,size)]

    Returns
    --------
    ptr: vector pointer of the RankSizePair*
    """
    cdef vector[RankSizePair*] *rsp_vec = new vector[RankSizePair*]()

    for idx, (rank, size) in enumerate(parts_to_sizes):
        rsp_vec.push_back(new RankSizePair(rank, size))

    return <uintptr_t>rsp_vec


def free_rank_size_pair(uintptr_t rank_size_ptr):
    """
    Function to free a vector of rankSizePair*

    Parameters
    ----------
    rank_size_ptr: pointer to a vector of rankSizePair*.
    """
    cdef vector[RankSizePair*] *rsp_vec = <vector[RankSizePair*]*>rank_size_ptr
    cdef RankSizePair *rsp_ptr

    for x_i in range(rsp_vec.size()):
        rsp_ptr = rsp_vec.at(x_i)
        del rsp_ptr
    del rsp_vec


def build_part_descriptor(m, n, uintptr_t rank_size_ptr, rank):
    """
    Function to build a shared PartDescriptor object.

    Parameters
    ----------
    m: int
        Total number of rows across all workers
    n: int
        Number of cols
    rank_size_ptr: int
        Pointer to a vector of RankSizePair*
    rank: int
        Rank to be mapped

    Returns
    --------
    ptr: int
        A pointer to a PartDescriptor
    """
    cdef vector[RankSizePair *] *rsp_vec = <vector[RankSizePair *]*>rank_size_ptr

    cdef PartDescriptor *descriptor = new PartDescriptor(
        <size_t>m,
        <size_t>n,
        <vector[RankSizePair*]>deref(rsp_vec),
        <int>rank,
    )
    return <uintptr_t>descriptor


def free_part_descriptor(uintptr_t descriptor_ptr):
    """
    Function to free a PartDescriptor*

    Parameters
    ----------
    descriptor_ptr: PartDescriptor* to be freed
    """
    cdef PartDescriptor *desc_c = <PartDescriptor*>descriptor_ptr
    del desc_c
