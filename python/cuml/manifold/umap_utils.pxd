#
# Copyright (c) 2022, NVIDIA CORPORATION.
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

# distutils: language = c++

from rmm._lib.memory_resource cimport DeviceMemoryResource
from rmm._lib.cuda_stream_view cimport cuda_stream_view
from libcpp.memory cimport unique_ptr

from libc.stdint cimport uint64_t, uintptr_t, int64_t
from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from cuml.metrics.distance_type cimport DistanceType

cdef extern from "cuml/manifold/umapparams.h" namespace "ML::UMAPParams":

    enum MetricType:
        EUCLIDEAN = 0,
        CATEGORICAL = 1

cdef extern from "cuml/common/callback.hpp" namespace "ML::Internals":

    cdef cppclass GraphBasedDimRedCallback

cdef extern from "cuml/manifold/umapparams.h" namespace "ML":

    cdef cppclass UMAPParams:
        int n_neighbors,
        int n_components,
        int n_epochs,
        float learning_rate,
        float min_dist,
        float spread,
        float set_op_mix_ratio,
        float local_connectivity,
        float repulsion_strength,
        int negative_sample_rate,
        float transform_queue_size,
        int verbosity,
        float a,
        float b,
        float initial_alpha,
        int init,
        int target_n_neighbors,
        MetricType target_metric,
        float target_weight,
        uint64_t random_state,
        bool deterministic,
        DistanceType metric,
        float p,
        GraphBasedDimRedCallback * callback

cdef extern from "raft/sparse/coo.hpp":
    cdef cppclass COO "raft::sparse::COO<float, int>":
        COO(cuda_stream_view stream)
        void allocate(int nnz, int size, bool init, cuda_stream_view stream)
        int nnz
        float* vals()
        int* rows()
        int* cols()

cdef class GraphHolder:
    cdef unique_ptr[COO] c_graph
    cdef DeviceMemoryResource mr

    @staticmethod
    cdef GraphHolder new_graph(cuda_stream_view stream)

    @staticmethod
    cdef GraphHolder from_ptr(unique_ptr[COO]& ptr)

    @staticmethod
    cdef GraphHolder from_coo_array(graph, handle, coo_array)

    cdef COO* get(GraphHolder self)
    cdef uintptr_t vals(GraphHolder self)
    cdef uintptr_t rows(GraphHolder self)
    cdef uintptr_t cols(GraphHolder self)
    cdef uint64_t get_nnz(GraphHolder self)
