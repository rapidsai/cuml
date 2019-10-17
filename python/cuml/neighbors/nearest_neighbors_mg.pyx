#
# Copyright (c) 2019, NVIDIA CORPORATION.
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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import numpy as np
import pandas as pd
import cudf
import ctypes
import cuml
import warnings

from cuml.common.base import Base
from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_dev_array, zeros, row_matrix

from cython.operator cimport dereference as deref

from libcpp cimport bool
from libcpp.memory cimport shared_ptr

import rmm
from libc.stdlib cimport malloc, free

from libc.stdint cimport uintptr_t, int64_t
from libc.stdlib cimport calloc, malloc, free

from numba import cuda
import rmm


class NearestNeighborsMG(NearestNeighbors):


    def _build_dataFloat(self, arr_interfaces):
        """
        Instantiate a container object for a float data pointer
        and size.
        :param arr_interfaces: 
        :return: 
        """
        cdef floatData_t ** dataF = < floatData_t ** > \
                                      malloc(sizeof(floatData_t *) \
                                             * len(arr_interfaces))
        cdef uintptr_t input_ptr
        for x_i in range(len(arr_interfaces)):
            x = arr_interfaces[x_i]
            input_ptr = x["data"]
            print("Shape: " + str(x["shape"]))
            dataF[x_i] = < floatData_t * > malloc(sizeof(floatData_t))
            dataF[x_i].ptr = < float * > input_ptr
            dataF[x_i].totalSize = < size_t > (x["shape"][0] * x["shape"][1] * sizeof(float))
            print("Size: " + str((x["shape"][0] * x["shape"][1] * sizeof(float))))

        return <size_t>dataF

    def _freeFloatD(self, data, arr_interfaces):
        cdef uintptr_t data_ptr = data
        cdef floatData_t **d = <floatData_t**>data_ptr
        for x_i in range(len(arr_interfaces)):
            free(d[x_i])
        free(d)

    def fit(self, X, partsToRanks):
        """
        Multi-node multi-GPU NearestNeighbors.

        NOTE: This implementation of NearestNeighbors is meant to be
        used with an initialized cumlCommunicator instance inside an
        existing distributed system. Refer to the Dask NearestNeighbors
         implementation in `cuml.dask.neighbors.nearest_neighbors`.

        :param X : cudf.Dataframe A dataframe to fit to the current model
        :return:
        """
        if self._should_downcast:
            warnings.warn("Parameter should_downcast is deprecated, use "
                          "convert_dtype in fit and kneighbors "
                          " methods instead. ")
            convert_dtype = True

        self.__del__()

        if len(X.shape) != 2:
            raise ValueError("data should be two dimensional")

        self.n_dims = X.shape[1]

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        cdef uintptr_t X_ctype = -1
        cdef uintptr_t dev_ptr = -1

        cdef float** input_arr
        cdef int* sizes_arr

        if isinstance(X, np.ndarray) and self.n_gpus > 1:

            if X.dtype != np.float32:
                if self._should_downcast or convert_dtype:
                    X = np.ascontiguousarray(X, np.float32)
                    if len(X[X == np.inf]) > 0:
                        raise ValueError("Downcast to single-precision "
                                         "resulted in data loss.")
                else:
                    raise TypeError("Only single precision floating point is"
                                    " supported for this algorithm. Use "
                                    "'convert_dtype=True' if you'd like it "
                                    "to be automatically casted to single "
                                    "precision.")

            sys_devices = set([d.id for d in cuda.gpus])

            if self.devices is not None:
                for d in self.devices:
                    if d not in sys_devices:
                        raise RuntimeError("Device %d is not available" % d)

                final_devices = self.devices

            else:
                n_gpus = min(self.n_gpus, len(sys_devices))
                final_devices = list(sys_devices)[:n_gpus]

            final_devices = np.ascontiguousarray(np.array(final_devices),
                                                 np.int32)

            X_ctype = X.ctypes.data
            dev_ptr = final_devices.ctypes.data

            input_arr = <float**> malloc(len(final_devices) * sizeof(float *))
            sizes_arr = <int*> malloc(len(final_devices) * sizeof(int))

            chunk_host_array(
                handle_[0],
                <float*>X_ctype,
                <int>X.shape[0],
                <int>X.shape[1],
                <int*>dev_ptr,
                <float**>input_arr,
                <int*>sizes_arr,
                <int>len(final_devices)
            )

            self.input = <size_t>input_arr
            self.sizes = <size_t>sizes_arr
            self.n_indices = len(final_devices)

        else:
            self.X_m, X_ctype, n_rows, n_cols, dtype = \
                input_to_dev_array(X, order='C', check_dtype=np.float32,
                                   convert_to_dtype=(np.float32
                                                     if convert_dtype
                                                     else None))

            input_arr = <float**> malloc(sizeof(float *))
            sizes_arr = <int*> malloc(sizeof(int))

            sizes_arr[0] = <int>len(X)
            input_arr[0] = <float*>X_ctype

            self.n_indices = 1

            self.input = <size_t>input_arr
            self.sizes = <size_t>sizes_arr

        return self

    def kneighbors(self, X):
