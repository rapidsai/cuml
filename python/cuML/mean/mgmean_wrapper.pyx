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

        n_rows, n_cols = gpu_alloc["shape"]
        gdf_datatype = np.dtype(gpu_alloc["dtype"])

        mean_ = pygdf.Series(np.zeros(n_cols, dtype=gdf_datatype))
        mean_gpu_array = mean_._column._data.to_gpu_array()

        cdef uintptr_t input_ptr = gpu_alloc["ptr"]
        cdef uintptr_t mean_ptr = mean_gpu_array.device_ctypes_pointer.value

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

        return mean_


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

        n_rows, n_cols = gpu_alloc["shape"]
        gdf_datatype = np.dtype(gpu_alloc["dtype"])

        mean_ = pygdf.Series(np.zeros(n_cols, dtype=gdf_datatype))
        mean_gpu_array = mean_._column._data.to_gpu_array()

        cdef uintptr_t input_ptr = gpu_alloc["ptr"]
        cdef uintptr_t mean_ptr = mean_gpu_array.device_ctypes_pointer.value

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

        return mean_

class MGMean:

    def _calc_float(self, gpu_allocs):

        n = len(gpu_allocs)
        mgd = DFloat(n)

        output = [mgd.build_mgd(gpu_alloc) for gpu_alloc in gpu_allocs]

        meanMG(
            < MGDescriptorFloat *> mgd.mean_data,
            < MGDescriptorFloat *> mgd.input_data,
            < int > n,
            < bool > False,
            < bool > False,
            < bool > False)

        return output

    def _calc_double(self, gpu_allocs):
        n = len(gpu_allocs)
        mgd = DDouble(n)
        output = [mgd.build_mgd(gpu_alloc) for gpu_alloc in gpu_allocs]

        meanMG(
            <MGDescriptorDouble *> mgd.mean_data,
            <MGDescriptorDouble *> mgd.input_data,
            < int > n,
            < bool > False,
            < bool > False,
            < bool > False)

        return output

    def calculate(self, gpu_allocs):

        """
        Calculate means on multiple GPUs at once. Eventually this will do the combining
        of the results using MPI in the C++ layer.
        :param gpu_allocs: A list of dicts containing the following keys: shape, dtype, ptr. Device 
                will be extracted automatically from the pointer. 
                
                TODO: This will need a hostname for MNMG.
        :return: 
        """

        gdf_datatype = np.dtype(gpu_allocs[0]["dtype"])

        if gdf_datatype == np.float32:
            output = self._calc_float(gpu_allocs)

        else:
            output = self._calc_double(gpu_allocs)

        return reduce(lambda x, y: x.__add__(y), output).__truediv__(len(output))