#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
from libc.stdint cimport int64_t, uint32_t, uint64_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from pylibraft.common.handle cimport handle_t
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.device_buffer cimport device_buffer

from cuml.internals.logger cimport level_enum
from cuml.metrics.distance_type cimport DistanceType


# Forward declaration for CAGRA index type
cdef extern from "cuvs/neighbors/cagra.hpp" namespace "cuvs::neighbors::cagra" nogil:
    cdef cppclass index[T, IdxT]:
        size_t size()
        uint32_t dim()
        uint32_t graph_degree()


# Type alias for the CAGRA index used in UMAP
cdef extern from "cuml/manifold/umap.hpp" namespace "ML" nogil:
    ctypedef index[float, uint32_t] cagra_index_t


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
        nn_descent_params_umap nnd "nn_descent_params"


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
    cdef cppclass COO "raft::sparse::COO<float, int, uint64_t>":
        COO(cuda_stream_view stream)
        void allocate(uint64_t nnz, int size, bool init, cuda_stream_view stream)
        uint64_t nnz
        float* vals()
        int* rows()
        int* cols()


cdef extern from "raft/core/host_coo_matrix.hpp" nogil:
    """
    class HostCOO : public raft::host_coo_matrix<float, int, int, uint64_t>
    {
        public:
            HostCOO()
                : raft::host_coo_matrix<float, int, int, uint64_t>(
                    raft::resources{}, 0, 0, 0) {}
            uint64_t get_nnz() {
                return this->structure_view().get_nnz();
            }

            int* rows() {
                return this->structure_view().get_rows().data();
            }

            int* cols() {
                return this->structure_view().get_cols().data();
            }

            float* vals() {
                return this->get_elements().data();
            }
    };
    """

    cdef cppclass HostCOO:
        HostCOO()
        uint64_t get_nnz()
        int* rows()
        int* cols()
        float* vals()


cdef extern from "cuml/manifold/umap.hpp" namespace "ML::UMAP" nogil:
    void fit(handle_t & handle,
             float * X,
             float * y,
             int n,
             int d,
             int64_t * knn_indices,
             float * knn_dists,
             UMAPParams * params,
             unique_ptr[device_buffer] & embeddings,
             HostCOO & graph,
             float * sigmas,
             float * rhos,
             unique_ptr[cagra_index_t] * cagra_index) except +

    void fit_sparse(handle_t &handle,
                    int *indptr,
                    int *indices,
                    float *data,
                    size_t nnz,
                    float *y,
                    int n,
                    int d,
                    int * knn_indices,
                    float * knn_dists,
                    UMAPParams *params,
                    unique_ptr[device_buffer] & embeddings,
                    HostCOO & graph) except +

    void transform(handle_t & handle,
                   float * X,
                   int n,
                   int d,
                   float * orig_X,
                   int orig_n,
                   float * embedding,
                   int embedding_n,
                   UMAPParams * params,
                   float * out,
                   cagra_index_t * cagra_index) except +

    void transform_sparse(handle_t &handle,
                          int *indptr,
                          int *indices,
                          float *data,
                          size_t nnz,
                          int n,
                          int d,
                          int *orig_x_indptr,
                          int *orig_x_indices,
                          float *orig_x_data,
                          size_t orig_nnz,
                          int orig_n,
                          float *embedding,
                          int embedding_n,
                          UMAPParams *params,
                          float *transformed) except +

    unique_ptr[COO] get_graph(handle_t &handle,
                              float* X,
                              float* y,
                              int n,
                              int d,
                              int64_t* knn_indices,
                              float* knn_dists,
                              UMAPParams* params) except +

    void refine(handle_t &handle,
                float* X,
                int n,
                int d,
                COO* cgraph_coo,
                UMAPParams* params,
                float* embeddings) except +

    void init_and_refine(handle_t &handle,
                         float* X,
                         int n,
                         int d,
                         COO* cgraph_coo,
                         UMAPParams* params,
                         float* embeddings) except +

    void inverse_transform(handle_t &handle,
                           float* inv_transformed,
                           int n,
                           int n_features,
                           float* orig_X,
                           int orig_n,
                           int* graph_rows,
                           int* graph_cols,
                           float* graph_vals,
                           int nnz,
                           float* sigmas,
                           float* rhos,
                           UMAPParams* params,
                           int n_epochs) except +
