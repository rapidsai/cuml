#
# SPDX-FileCopyrightText: Copyright (c) 2019, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

# distutils: language = c++

from libc.stdint cimport uintptr_t


cdef extern from "ml_cuda_utils.h" namespace "ML":
    cdef int get_device(void *ptr) except +


def device_of_gpu_matrix(g):
    cdef uintptr_t cptr = g.device_ctypes_pointer.value
    return get_device(<void*> cptr)
