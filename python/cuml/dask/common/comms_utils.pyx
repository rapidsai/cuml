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

from cuml.common.handle import Handle

from cpython cimport array
import array

import dask.distributed
from libcpp cimport bool
from libc.stdlib cimport malloc, free
import re
import os
from cython.operator cimport dereference as deref

import numpy as np
import numba.cuda
import cudf

from libc.stdint cimport uintptr_t

import cudf

cdef extern from "common/cuML_comms_impl.cpp" namespace "MLCommon":
    cdef cppclass cumlCommunicator


cdef extern from "cuML.hpp" namespace "ML" nogil:
    cdef cppclass cumlHandle:
        cumlHandle() except +

cdef extern from "cuML_comms.hpp" namespace "ML":
    void inject_comms_py(cumlHandle *handle, ncclComm_t comm, void *ucp_worker, void *eps, int size, int rank)


def inject_comms_on_handle(handle, nccl_inst, ucp_worker, eps, size, rank):

    cdef size_t *ucp_eps = <size_t*> malloc(len(eps)*sizeof(size_t))

    cdef size_t ep_st
    for i in range(len(eps)):
        if eps[i] is not None:
            ucp_eps[i] = <size_t>eps[i].get_ep()
            test_ep(<void*><size_t>eps[i].get_ep())
        else:
            ucp_eps[i] = 0

    cdef void* ucp_worker_st = <void*><size_t>ucp_worker

    cdef size_t handle_size_t = <size_t>handle.getHandle()
    handle_ = <cumlHandle*>handle_size_t

    cdef size_t nccl_comm_size_t = <size_t>nccl_inst.get_comm()
    nccl_comm_ = <ncclComm_t*>nccl_comm_size_t

    inject_comms_py(handle_, deref(nccl_comm_), <void*>ucp_worker_st, <void*>ucp_eps, size, rank)
