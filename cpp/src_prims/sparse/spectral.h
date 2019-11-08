/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <nvgraph.h>

#include "sparse/nvgraph_wrappers.h"

#include "selection/knn.h"
#include "sparse/coo.h"

#include <cuml/common/cuml_allocator.hpp>
#include "common/device_buffer.hpp"

#include "cuda_utils.h"

namespace MLCommon {
namespace Spectral {

template <typename T>
void fit_clusters(int *rows, int *cols, T *vals, int nnz, int n, int n_clusters,
                  float eigen_tol, int *out,
                  std::shared_ptr<deviceAllocator> allocator,
                  cudaStream_t stream) {
  nvgraphHandle_t graphHandle;
  cudaDataType_t edge_dimT = CUDA_R_32F;
  NVGRAPH_CHECK(nvgraphCreate(&graphHandle));

  // Allocate csr arrays
  device_buffer<int> src_offsets(allocator, stream, n + 1);
  device_buffer<int> dst_indices(allocator, stream, nnz);

  nvgraphCOOTopology32I_st *COO_input = new nvgraphCOOTopology32I_st();
  COO_input->nedges = nnz;
  COO_input->nvertices = n;
  COO_input->source_indices = rows;
  COO_input->destination_indices = cols;

  nvgraphCSRTopology32I_st *CSR_input = new nvgraphCSRTopology32I_st();
  CSR_input->destination_indices = dst_indices.data();
  CSR_input->nedges = nnz;
  CSR_input->nvertices = n;
  CSR_input->source_offsets = src_offsets.data();

  NVGRAPH_CHECK(nvgraphConvertTopology(
    graphHandle, NVGRAPH_COO_32, (void *)COO_input, (void *)vals, &edge_dimT,
    NVGRAPH_CSR_32, (void *)CSR_input, (void *)vals));

  int weight_index = 0;

  device_buffer<T> eigVals(allocator, stream, n_clusters);
  device_buffer<T> embedding(allocator, stream, n * n_clusters);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Spectral clustering parameters
  struct SpectralClusteringParameter clustering_params;
  clustering_params.n_clusters = n_clusters;
  clustering_params.n_eig_vects = n_clusters;
  clustering_params.algorithm = NVGRAPH_BALANCED_CUT_LANCZOS;
  clustering_params.evs_tolerance = eigen_tol;
  clustering_params.evs_max_iter = 0;
  clustering_params.kmean_tolerance = 0.0f;
  clustering_params.kmean_max_iter = 0;

  nvgraphGraphDescr_t graph;
  NVGRAPH_CHECK(nvgraphCreateGraphDescr(graphHandle, &graph));
  NVGRAPH_CHECK(nvgraphSetGraphStructure(graphHandle, graph, (void *)CSR_input,
                                         NVGRAPH_CSR_32));
  NVGRAPH_CHECK(nvgraphAllocateEdgeData(graphHandle, graph, 1, &edge_dimT));
  NVGRAPH_CHECK(nvgraphSetEdgeData(graphHandle, graph, (void *)vals, 0));

  NVGRAPH_CHECK(nvgraphSpectralClustering(graphHandle, graph, weight_index,
                                          &clustering_params, out,
                                          eigVals.data(), embedding.data()));

  NVGRAPH_CHECK(nvgraphDestroyGraphDescr(graphHandle, graph));
  NVGRAPH_CHECK(nvgraphDestroy(graphHandle));

  free(COO_input);
  free(CSR_input);
}

template <typename T>
void fit_clusters(long *knn_indices, T *knn_dists, int m, int n_neighbors,
                  int n_clusters, float eigen_tol, int *out,
                  std::shared_ptr<deviceAllocator> allocator,
                  cudaStream_t stream) {
  device_buffer<int> rows(allocator, stream, m * n_neighbors);
  device_buffer<int> cols(allocator, stream, m * n_neighbors);
  device_buffer<T> vals(allocator, stream, m * n_neighbors);

  MLCommon::Sparse::from_knn(knn_indices, knn_dists, m, n_neighbors,
                             rows.data(), cols.data(), vals.data());

  // todo: might need to symmetrize the knn to create the knn graph

  fit_clusters(rows.data(), cols.data(), vals.data(), m * n_neighbors, m,
               n_clusters, eigen_tol, out, allocator, stream);
}

template <typename T>
void fit_clusters(T *X, int m, int n, int n_neighbors, int n_clusters,
                  float eigen_tol, int *out,
                  std::shared_ptr<deviceAllocator> allocator,
                  cudaStream_t stream) {
  device_buffer<int64_t> knn_indices(allocator, stream, m * n_neighbors);
  device_buffer<float> knn_dists(allocator, stream, m * n_neighbors);

  float **ptrs = new float *[1];
  int *sizes = new int[1];
  ptrs[0] = X;
  sizes[0] = m;

  MLCommon::Selection::brute_force_knn(ptrs, sizes, 1, n, X, m,
                                       knn_indices.data(), knn_dists.data(),
                                       n_neighbors, allocator, stream);

  fit_clusters(knn_indices.data(), knn_dists.data(), m, n_neighbors, n_clusters,
               eigen_tol, out, allocator, stream);

  delete ptrs;
  delete sizes;
}

template <typename T>
void fit_embedding(int *rows, int *cols, T *vals, int nnz, int n,
                   int n_components, T *out,
                   std::shared_ptr<deviceAllocator> allocator,
                   cudaStream_t stream) {
  nvgraphHandle_t grapHandle;
  cudaDataType_t edge_dimT = CUDA_R_32F;
  NVGRAPH_CHECK(nvgraphCreate(&grapHandle));

  device_buffer<int> src_offsets(allocator, stream, n + 1);
  device_buffer<int> dst_indices(allocator, stream, nnz);

  nvgraphCOOTopology32I_st *COO_input = new nvgraphCOOTopology32I_st();
  COO_input->nedges = nnz;
  COO_input->nvertices = n;
  COO_input->source_indices = rows;
  COO_input->destination_indices = cols;

  nvgraphCSRTopology32I_st *CSR_input = new nvgraphCSRTopology32I_st();
  CSR_input->destination_indices = dst_indices.data();
  CSR_input->nedges = nnz;
  CSR_input->nvertices = n;
  CSR_input->source_offsets = src_offsets.data();

  NVGRAPH_CHECK(nvgraphConvertTopology(
    grapHandle, NVGRAPH_COO_32, (void *)COO_input, (void *)vals, &edge_dimT,
    NVGRAPH_CSR_32, (void *)CSR_input, (void *)vals));

  int weight_index = 0;

  device_buffer<T> eigVals(allocator, stream, n_components + 1);
  device_buffer<T> eigVecs(allocator, stream, n * (n_components + 1));
  device_buffer<int> labels(allocator, stream, n);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Spectral clustering parameters
  struct SpectralClusteringParameter clustering_params;
  clustering_params.n_clusters = n_components + 1;
  clustering_params.n_eig_vects = n_components + 1;
  clustering_params.algorithm = NVGRAPH_BALANCED_CUT_LANCZOS;
  clustering_params.evs_tolerance = 0.01f;
  clustering_params.evs_max_iter = 0;
  clustering_params.kmean_tolerance = 0.0f;
  clustering_params.kmean_max_iter = 1;

  nvgraphGraphDescr_t graph;
  NVGRAPH_CHECK(nvgraphCreateGraphDescr(grapHandle, &graph));
  NVGRAPH_CHECK(nvgraphSetGraphStructure(grapHandle, graph, (void *)CSR_input,
                                         NVGRAPH_CSR_32));
  NVGRAPH_CHECK(nvgraphAllocateEdgeData(grapHandle, graph, 1, &edge_dimT));
  NVGRAPH_CHECK(nvgraphSetEdgeData(grapHandle, graph, (void *)vals, 0));

  NVGRAPH_CHECK(nvgraphSpectralClustering(grapHandle, graph, weight_index,
                                          &clustering_params, labels.data(),
                                          eigVals.data(), eigVecs.data()));

  NVGRAPH_CHECK(nvgraphDestroyGraphDescr(grapHandle, graph));
  NVGRAPH_CHECK(nvgraphDestroy(grapHandle));

  MLCommon::copy<T>(out, eigVecs.data() + n, n * n_components, stream);

  CUDA_CHECK(cudaPeekAtLastError());

  free(COO_input);
  free(CSR_input);
}

template <typename T>
void fit_embedding(long *knn_indices, float *knn_dists, int m, int n_neighbors,
                   int n_components, T *out,
                   std::shared_ptr<deviceAllocator> allocator,
                   cudaStream_t stream) {
  device_buffer<int> rows(allocator, stream, m * n_neighbors);
  device_buffer<int> cols(allocator, stream, m * n_neighbors);
  device_buffer<T> vals(allocator, stream, m * n_neighbors);

  MLCommon::Sparse::from_knn(knn_indices, knn_dists, m, n_neighbors,
                             rows.data(), cols.data(), vals.data());

  // todo: might need to symmetrize the knn graph here. UMAP works here because
  // it has already done this.

  fit_embedding(rows.data(), cols.data(), vals.data(), m * n_neighbors, m,
                n_components, out, allocator, stream);
}

template <typename T>
void fit_embedding(T *X, int m, int n, int n_neighbors, int n_components,
                   T *out, std::shared_ptr<deviceAllocator> allocator,
                   cudaStream_t stream) {
  device_buffer<int64_t> knn_indices(allocator, stream, m * n_neighbors);
  device_buffer<float> knn_dists(allocator, stream, m * n_neighbors);

  float **ptrs = new float *[1];
  int *sizes = new int[1];
  ptrs[0] = X;
  sizes[0] = m;

  MLCommon::Selection::brute_force_knn(ptrs, sizes, 1, n, X, m,
                                       knn_indices.data(), knn_dists.data(),
                                       n_neighbors, allocator, stream);

  fit_embedding(knn_indices.data(), knn_dists.data(), m, n_neighbors,
                n_components, out, allocator, stream);

  delete ptrs;
  delete sizes;
}
}  // namespace Spectral
}  // namespace MLCommon
