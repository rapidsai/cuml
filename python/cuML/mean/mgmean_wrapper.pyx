# Copyright (c) 2018, NVIDIA CORPORATION.
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

from libc.stdlib cimport malloc, free
from cython.operator cimport dereference as deref

from functools import reduce

cimport numpy as np
from numba import cuda
import cudf as pygdf
from libcpp cimport bool
from libc.stdint cimport uintptr_t
import ctypes
from c_mean cimport *

cdef class DFloat(object):

    """
    Container class to maintain C++ object references
    """
    cdef MGDescriptorFloat * input_data
    cdef MGDescriptorFloat * mean_data

    cdef int i
    cdef int n

    def __cinit__(self, n):
        self.input_data = < MGDescriptorFloat * > malloc(n * sizeof(MGDescriptorFloat))
        self.mean_data =  < MGDescriptorFloat *> malloc(n * sizeof(MGDescriptorFloat))

        self.i = 0

    def __init__(self, n):
        self.n = n

    def build_mgd(self, gpu_alloc):

        in_alloc, out_alloc = gpu_alloc
        n_rows, n_cols = in_alloc["shape"]

        cdef uintptr_t input_ptr = in_alloc["data"][0].value
        cdef uintptr_t mean_ptr = out_alloc["data"][0].value

        input_mgd = new MGDescriptorFloat(
            < float * > input_ptr,
            < int > n_rows,
            < int > n_cols
        )

        mean_mgd = new MGDescriptorFloat(
            < float * > mean_ptr,
            < int > 1,
            < int > n_cols
        )

        self.input_data[self.i] = deref(input_mgd)
        self.mean_data[self.i] = deref(mean_mgd)
        self.i += 1


cdef class DDouble(object):

    """
    Container class to maintain C++ object references
    """
    cdef MGDescriptorDouble * input_data
    cdef MGDescriptorDouble * mean_data

    cdef int i
    cdef int n

    def __cinit__(self, n):
        self.input_data = <MGDescriptorDouble *> malloc(n * sizeof(MGDescriptorDouble))
        self.mean_data =  <MGDescriptorDouble *> malloc(n * sizeof(MGDescriptorDouble))

        self.i = 0

    def __init__(self, n):
        self.n = n

    def build_mgd(self, gpu_alloc):

        in_alloc, out_alloc = gpu_alloc
        n_rows, n_cols = in_alloc["shape"]

        cdef uintptr_t input_ptr = in_alloc["data"].value
        cdef uintptr_t mean_ptr = out_alloc["data"].value

        input_mgd = new MGDescriptorDouble(
            < double * > input_ptr,
            < int > n_rows,
            < int > n_cols
        )

        mean_mgd = new MGDescriptorDouble(
            < double * > mean_ptr,
            < int > 1,
            < int > n_cols
        )

        self.input_data[self.i] = deref(input_mgd)
        self.mean_data[self.i] = deref(mean_mgd)

        self.i += 1

class MGMean:

    def _calc_float(self, gpu_allocs):

        n = len(gpu_allocs)
        mgd = DFloat(n)
        [mgd.build_mgd(gpu_alloc) for gpu_alloc in gpu_allocs]

        meanMG(
            < MGDescriptorFloat *> mgd.mean_data,
            < MGDescriptorFloat *> mgd.input_data,
            < int > n,
            < bool > False,
            < bool > False,
            < bool > False)


    def _calc_double(self, gpu_allocs):
        n = len(gpu_allocs)
        mgd = DDouble(n)
        [mgd.build_mgd(gpu_alloc) for gpu_alloc in gpu_allocs]

        meanMG(
            <MGDescriptorDouble *> mgd.mean_data,
            <MGDescriptorDouble *> mgd.input_data,
            < int > n,
            < bool > False,
            < bool > False,
            < bool > False)

    def calculate(self, gpu_allocs):

        """
        Calculate means on multiple GPUs at once. Eventually this will do the combining
        of the results using MPI in the C++ layer.
        :param gpu_allocs: A list of dicts containing the following keys: shape, dtype, ptr. Device 
                will be extracted automatically from the pointer. 
        :return:
        """

        # Pull the first alloc to determine the type (all should be the same)
        in_alloc, out_alloc = gpu_allocs[0]

        gdf_datatype = np.dtype(in_alloc["typestr"])

        if gdf_datatype == np.float32:
            self._calc_float(gpu_allocs)

        else:
            self._calc_double(gpu_allocs)