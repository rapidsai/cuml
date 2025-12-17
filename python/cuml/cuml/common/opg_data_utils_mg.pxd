#
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

# Util functions, will be moved to their own file as the other methods are
# refactored
# todo: use cuda_array_interface instead of arr_interfaces for building this

from libc.stdint cimport int64_t
from libcpp.vector cimport vector


cdef extern from "opg/matrix/data.hpp" namespace "MLCommon::Matrix":
    cdef cppclass Data[T]:
        Data(T *ptr, size_t totalSize)

    cdef cppclass floatData_t:
        floatData_t(float *ptr, size_t totalSize)
        float *ptr
        size_t totalSize

    cdef cppclass doubleData_t:
        doubleData_t(double *ptr, size_t totalSize)
        double *ptr
        size_t totalSize

ctypedef Data[int64_t] int64Data_t
ctypedef Data[int] intData_t
ctypedef vector[int*] int_ptr_vector
ctypedef vector[float*] float_ptr_vector

cdef extern from "opg/matrix/part_descriptor.hpp" \
                 namespace "MLCommon::Matrix":

    cdef cppclass RankSizePair:
        int rank
        size_t size

    cdef cppclass PartDescriptor:
        PartDescriptor(size_t M,
                       size_t N,
                       vector[RankSizePair *] &partsToRanks,
                       int myrank)
