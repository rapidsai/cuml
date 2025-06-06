#
# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

from libc.stdint cimport int64_t, uint64_t, uintptr_t
from libcpp cimport bool
from libcpp.memory cimport shared_ptr, unique_ptr
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from cuml.internals.logger cimport level_enum
from cuml.metrics.distance_type cimport DistanceType


cdef extern from "cuml/manifold/umapparams.h" namespace "ML::UMAPParams" nogil:

    enum MetricType:
        EUCLIDEAN = 0,
        CATEGORICAL = 1
    enum graph_build_algo:
        BRUTE_FORCE_KNN = 0,
        NN_DESCENT = 1

cdef extern from "cuml/common/callback.hpp" namespace "ML::Internals":

    cdef cppclass GraphBasedDimRedCallback

cdef extern from "cuml/manifold/umapparams.h" namespace "graph_build_params" nogil:
    cdef cppclass nn_descent_params_umap:
        size_t graph_degree
        size_t intermediate_graph_degree
        size_t max_iterations
        float termination_threshold

    cdef cppclass graph_build_params:
        size_t overlap_factor
        size_t n_clusters
        nn_descent_params_umap nn_descent_params

cdef extern from "cuml/manifold/umapparams.h" namespace "ML" nogil:

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
        level_enum verbosity,
        float a,
        float b,
        float initial_alpha,
        int init,
        graph_build_algo build_algo,
        graph_build_params build_params,
        int target_n_neighbors,
        MetricType target_metric,
        float target_weight,
        uint64_t random_state,
        bool deterministic,
        DistanceType metric,
        float p,
        GraphBasedDimRedCallback * callback,

cdef extern from "raft/sparse/coo.hpp" nogil:
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
