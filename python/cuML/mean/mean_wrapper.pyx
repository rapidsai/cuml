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

cimport c_mean
import numpy as np

cimport numpy as np
from numba import cuda
import cudf as pygdf
from libcpp cimport bool
from libc.stdint cimport uintptr_t
import ctypes
from c_mean cimport *

class Mean:

    def _get_ctype_ptr(self, obj):
        # The manner to access the pointers in the gdf's might change, so
        # encapsulating access in the following 3 methods. They might also be
        # part of future gdf versions.
        return obj.device_ctypes_pointer.value

    def _get_column_ptr(self, obj):
        return self._get_ctype_ptr(obj._column._data.to_gpu_array())

    def _get_gdf_as_matrix_ptr(self, gdf):
        return self._get_ctype_ptr(gdf.as_gpu_matrix(order='C'))

    def calculate(self, input_gdf):

        # python params
        self.n_rows = len(input_gdf)
        self.n_cols = len(input_gdf._cols)

        x = []
        for col in input_gdf.columns:
            x.append(input_gdf[col]._column.dtype)
            break

        self.gdf_datatype = np.dtype(x[0])
        self.mean_ = pygdf.Series(np.zeros(self.n_cols, dtype=self.gdf_datatype))

        gpu_data = input_gdf.as_gpu_matrix()

        cdef uintptr_t input_ptr = self._get_ctype_ptr(gpu_data)
        cdef uintptr_t mean_ptr = self._get_column_ptr(self.mean_)

        import sys
        print("Python Allocated memory pointer: " + hex(mean_ptr))
        print(
            "Python Allocated memory size (calculated): " + str(self.n_cols * self.gdf_datatype.itemsize * self.n_rows))
        print("Python Allocated memory size (actual): " + str(input_gdf.as_gpu_matrix().alloc_size))
        print("Python Data type size: " + str(self.gdf_datatype.itemsize))
        print("Python Allocated ending pointer: " + hex(
            mean_ptr + (self.n_cols * self.gdf_datatype.itemsize * self.n_rows)))
        print("Python Allocated memory pointer (input): " + hex(input_ptr))
        print("Numba info: " + str(input_gdf.as_gpu_matrix().__cuda_array_interface__))
        print("Numba memory: " + str(input_gdf.as_gpu_matrix().__cuda_memory__))

        if self.gdf_datatype.type == np.float32:
                c_mean.mean( < float * > mean_ptr,
                < float * > input_ptr,
                < int > self.n_cols,
                < int > self.n_rows,
                < bool > True,
                < bool > False)
        else:
            c_mean.mean( < double * > mean_ptr,
                < double * > input_ptr,
                < int > self.n_cols,
                < int > self.n_rows,
                < bool > True,
                < bool > False)

        return self.mean_

        # def __init__(self, input_gdf):
        # return calculate(input_gdf);
