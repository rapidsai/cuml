/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include "common/device_buffer.hpp"
#include "cuda_utils.h"
#include "cuml/common/cuml_allocator.hpp"
#include "cusparse_wrappers.h"
#include "selection/knn.h"
#include "sparse/coo.h"

namespace MLCommon {
namespace Spectral {

#define NVGRAPH_CHECK(call)                                                    \
  do {                                                                         \
    nvgraphStatus_t status = call;                                             \
    ASSERT(status == NVGRAPH_STATUS_SUCCESS, "FAIL: call='%s', status='%d'\n", \
           #call, status);                                                     \
  } while (0)

template <typename T>
void coo2csr(cusparseHandle_t handle, const int *srcRows, const int *srcCols,
             const T *srcVals, int nnz, int m, int *dst_offsets, int *dstCols,
             T *dstVals, std::shared_ptr<deviceAllocator> d_alloc,
             cudaStream_t stream) {
  device_buffer<int> dstRows(d_alloc, stream, nnz);
  CUDA_CHECK(cudaMemcpyAsync(dstRows.data(), srcRows, sizeof(int) * nnz,
                             cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(dstCols, srcCols, sizeof(int) * nnz,
                             cudaMemcpyDeviceToDevice, stream));
  auto buffSize = Sparse::cusparsecoosort_bufferSizeExt(
    handle, m, m, nnz, srcRows, srcCols, stream);
  device_buffer<char> pBuffer(d_alloc, stream, buffSize);
  device_buffer<int> P(d_alloc, stream, nnz);
  CUSPARSE_CHECK(cusparseCreateIdentityPermutation(handle, nnz, P.data()));
  Sparse::cusparsecoosortByRow(handle, m, m, nnz, dstRows.data(), dstCols,
                               P.data(), pBuffer.data(), stream);
  Sparse::cusparsegthr(handle, nnz, srcVals, dstVals, P.data(), stream);
  Sparse::cusparsecoo2csr(handle, dstRows.data(), nnz, m, dst_offsets, stream);
  CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename T>
void fit_clusters(cusparseHandle_t handle, int *rows, int *cols, T *vals,
                  int nnz, int n, int n_clusters, float eigen_tol, int *out,
                  std::shared_ptr<deviceAllocator> d_alloc,
                  cudaStream_t stream) {
  nvgraphHandle_t graphHandle;
  cudaDataType_t edge_dimT = CUDA_R_32F;
  NVGRAPH_CHECK(nvgraphCreate(&graphHandle));

  // conversion
  device_buffer<int> src_offsets(d_alloc, stream, n + 1);
  device_buffer<int> dst_cols(d_alloc, stream, nnz);
  device_buffer<T> dst_vals(d_alloc, stream, nnz);
  coo2csr(handle, rows, cols, vals, nnz, n, src_offsets.data(), dst_cols.data(),
          dst_vals.data(), d_alloc, stream);

  nvgraphCSRTopology32I_st CSR_input;
  CSR_input.destination_indices = dst_cols.data();
  CSR_input.nedges = nnz;
  CSR_input.nvertices = n;
  CSR_input.source_offsets = src_offsets.data();

  int weight_index = 0;

  device_buffer<T> eigVals(d_alloc, stream, n_clusters);
  device_buffer<T> embedding(d_alloc, stream, n * n_clusters);

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
  NVGRAPH_CHECK(nvgraphSetGraphStructure(graphHandle, graph, (void *)&CSR_input,
                                         NVGRAPH_CSR_32));
  NVGRAPH_CHECK(nvgraphAllocateEdgeData(graphHandle, graph, 1, &edge_dimT));
  NVGRAPH_CHECK(
    nvgraphSetEdgeData(graphHandle, graph, (void *)dst_vals.data(), 0));

  NVGRAPH_CHECK(nvgraphSpectralClustering(graphHandle, graph, weight_index,
                                          &clustering_params, out,
                                          eigVals.data(), embedding.data()));

  NVGRAPH_CHECK(nvgraphDestroyGraphDescr(graphHandle, graph));
  NVGRAPH_CHECK(nvgraphDestroy(graphHandle));
}

template <typename T>
void fit_clusters(cusparseHandle_t handle, long *knn_indices, T *knn_dists,
                  int m, int n_neighbors, int n_clusters, float eigen_tol,
                  int *out, std::shared_ptr<deviceAllocator> d_alloc,
                  cudaStream_t stream) {
  device_buffer<int> rows(d_alloc, stream, m * n_neighbors);
  device_buffer<int> cols(d_alloc, stream, m * n_neighbors);
  device_buffer<T> vals(d_alloc, stream, m * n_neighbors);
  MLCommon::Sparse::from_knn(knn_indices, knn_dists, m, n_neighbors,
                             rows.data(), cols.data(), vals.data());
  fit_clusters(handle, rows.data(), cols.data(), vals.data(), m * n_neighbors,
               m, n_clusters, eigen_tol, out, d_alloc, stream);
}

template <typename T>
void fit_clusters(cusparseHandle_t handle, T *X, int m, int n, int n_neighbors,
                  int n_clusters, float eigen_tol, int *out,
                  std::shared_ptr<deviceAllocator> d_alloc,
                  cudaStream_t stream) {
  device_buffer<long> knn_indices(d_alloc, stream, m * n_neighbors);
  device_buffer<float> knn_dists(d_alloc, stream, m * n_neighbors);
  std::vector<float *> ptrs(1);
  std::vector<int> sizes(1);
  ptrs[0] = X;
  sizes[0] = m;
  MLCommon::Selection::brute_force_knn(ptrs, sizes, n, X, m, knn_indices.data(),
                                       knn_dists.data(), n_neighbors, d_alloc,
                                       stream);
  fit_clusters(handle, knn_indices.data(), knn_dists.data(), m, n_neighbors,
               n_clusters, eigen_tol, out, d_alloc, stream);
}

template <typename T>
void fit_embedding(cusparseHandle_t handle, int *rows, int *cols, T *vals,
                   int nnz, int n, int n_components, T *out,
                   std::shared_ptr<deviceAllocator> d_alloc,
                   cudaStream_t stream) {
  nvgraphHandle_t grapHandle;
  cudaDataType_t edge_dimT = CUDA_R_32F;
  NVGRAPH_CHECK(nvgraphCreate(&grapHandle));

  device_buffer<int> src_offsets(d_alloc, stream, n + 1);
  device_buffer<int> dst_cols(d_alloc, stream, nnz);
  device_buffer<T> dst_vals(d_alloc, stream, nnz);
  coo2csr(handle, rows, cols, vals, nnz, n, src_offsets.data(), dst_cols.data(),
          dst_vals.data(), d_alloc, stream);

  nvgraphCSRTopology32I_st CSR_input;
  CSR_input.destination_indices = dst_cols.data();
  CSR_input.nedges = nnz;
  CSR_input.nvertices = n;
  CSR_input.source_offsets = src_offsets.data();

  int weight_index = 0;

  device_buffer<T> eigVals(d_alloc, stream, n_components + 1);
  device_buffer<T> eigVecs(d_alloc, stream, n * (n_components + 1));
  device_buffer<int> labels(d_alloc, stream, n);

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
  NVGRAPH_CHECK(nvgraphSetGraphStructure(grapHandle, graph, (void *)&CSR_input,
                                         NVGRAPH_CSR_32));
  NVGRAPH_CHECK(nvgraphAllocateEdgeData(grapHandle, graph, 1, &edge_dimT));
  NVGRAPH_CHECK(nvgraphSetEdgeData(grapHandle, graph, (void *)vals, 0));

  NVGRAPH_CHECK(nvgraphSpectralClustering(grapHandle, graph, weight_index,
                                          &clustering_params, labels.data(),
                                          eigVals.data(), eigVecs.data()));

  NVGRAPH_CHECK(nvgraphDestroyGraphDescr(grapHandle, graph));
  NVGRAPH_CHECK(nvgraphDestroy(grapHandle));

  MLCommon::copy<T>(out, eigVecs.data() + n, n * n_components, stream);

  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void fit_embedding(cusparseHandle_t handle, long *knn_indices, float *knn_dists,
                   int m, int n_neighbors, int n_components, T *out,
                   std::shared_ptr<deviceAllocator> d_alloc,
                   cudaStream_t stream) {
  device_buffer<int> rows(d_alloc, stream, m * n_neighbors);
  device_buffer<int> cols(d_alloc, stream, m * n_neighbors);
  device_buffer<T> vals(d_alloc, stream, m * n_neighbors);

  MLCommon::Sparse::from_knn(knn_indices, knn_dists, m, n_neighbors,
                             rows.data(), cols.data(), vals.data());

  fit_embedding(handle, rows.data(), cols.data(), vals.data(), m * n_neighbors,
                m, n_components, out, d_alloc, stream);
}

template <typename T>
void fit_embedding(cusparseHandle_t handle, T *X, int m, int n, int n_neighbors,
                   int n_components, T *out,
                   std::shared_ptr<deviceAllocator> d_alloc,
                   cudaStream_t stream) {
  device_buffer<int64_t> knn_indices(d_alloc, stream, m * n_neighbors);
  device_buffer<float> knn_dists(d_alloc, stream, m * n_neighbors);

  std::vector<float *> ptrs(1);
  std::vector<int> sizes(1);
  ptrs[0] = X;
  sizes[0] = m;

  MLCommon::Selection::brute_force_knn(ptrs, sizes, n, X, m, knn_indices.data(),
                                       knn_dists.data(), n_neighbors, d_alloc,
                                       stream);

  fit_embedding(handle, knn_indices.data(), knn_dists.data(), m, n_neighbors,
                n_components, out, d_alloc, stream);
}
}  // namespace Spectral
}  // namespace MLCommon
