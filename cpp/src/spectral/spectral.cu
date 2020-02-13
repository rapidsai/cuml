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

#include <cuml/cuml.hpp>
#include "sparse/coo.h"

#include "sparse/spectral.h"

namespace ML {

namespace Spectral {

/***
   * Given a (symmetric) knn graph in COO format, this function computes the spectral
   * clustering, using Lanczos min cut algorithm and k-means.
   * @param rows source vertices of knn graph
   * @param cols destination vertices of knn graph
   * @param vals edge weights (distances) connecting source & destination vertices
   * @param nnz number of nonzero edge weights in vals (size of rows/cols/vals)
   * @param n total number of vertices in graph (n_samples)
   * @param n_clusters the number of clusters to fit
   * @param eigen_tol the tolerance threshold for the eigensolver
   * @param out output array for labels (size m)
   */

void fit_clusters(const cumlHandle &handle, int *rows, int *cols, float *vals,
                  int nnz, int n, int n_clusters, float eigen_tol, int *out) {
  const auto &impl = handle.getImpl();
  MLCommon::Spectral::fit_clusters(
    impl.getcusparseHandle(), rows, cols, vals, nnz, n, n_clusters, eigen_tol,
    out, handle.getDeviceAllocator(), handle.getStream());
}

/***
   * Given a indices and distances matrices, this function computes the spectral
   * clustering, using Lanczos min cut algorithm and k-means.
   * @param knn_indices m*n_neighbors matrix of nearest indices
   * @param knn_dists m*n_neighbors matrix of distances to nearest neigbors
   * @param m number of vertices in knn_indices and knn_dists
   * @param n_neighbors the number of neighbors to query for knn graph construction
   * @param n_clusters the number of clusters to fit
   * @param eigen_tol the tolerance threshold for the eigensolver
   * @param out output array for labels (size m)
   */
void fit_clusters(const cumlHandle &handle, long *knn_indices, float *knn_dists,
                  int m, int n_neighbors, int n_clusters, float eigen_tol,
                  int *out) {
  const auto &impl = handle.getImpl();
  MLCommon::Spectral::fit_clusters(impl.getcusparseHandle(), knn_indices,
                                   knn_dists, m, n_neighbors, n_clusters,
                                   eigen_tol, out, handle.getDeviceAllocator(),
                                   handle.getStream());
}

/***
   * Given a feature matrix, this function computes the spectral
   * clustering, using Lanczos min cut algorithm and k-means.
   * @param X a feature matrix (size m*n)
   * @param m number of samples in X
   * @param n number of features in X
   * @param n_neighbors the number of neighbors to query for knn graph construction
   * @param n_clusters the number of clusters to fit
   * @param eigen_tol the tolerance threshold for the eigensolver
   * @param out output array for labels (size m)
   */
void fit_clusters(const cumlHandle &handle, float *X, int m, int n,
                  int n_neighbors, int n_clusters, float eigen_tol, int *out) {
  const auto &impl = handle.getImpl();
  MLCommon::Spectral::fit_clusters(
    impl.getcusparseHandle(), X, m, n, n_neighbors, n_clusters, eigen_tol, out,
    handle.getDeviceAllocator(), handle.getStream());
}

/**
   * Given a COO formatted (symmetric) knn graph, this function
   * computes the spectral embeddings (lowest n_components
   * eigenvectors), using Lanczos min cut algorithm.
   * @param rows source vertices of knn graph (size nnz)
   * @param cols destination vertices of knn graph (size nnz)
   * @param vals edge weights connecting vertices of knn graph (size nnz)
   * @param nnz size of rows/cols/vals
   * @param n number of samples in X
   * @param n_neighbors the number of neighbors to query for knn graph construction
   * @param n_components the number of components to project the X into
   * @param out output array for embedding (size n*n_comonents)
   */
void fit_embedding(const cumlHandle &handle, int *rows, int *cols, float *vals,
                   int nnz, int n, int n_components, float *out) {
  const auto &impl = handle.getImpl();
  MLCommon::Spectral::fit_embedding(
    impl.getcusparseHandle(), rows, cols, vals, nnz, n, n_components, out,
    handle.getDeviceAllocator(), handle.getStream());
}

/***
   * Given index and distance matrices returned from a knn query, this
   * function computes the spectral embeddings (lowest n_components
   * eigenvectors), using Lanczos min cut algorithm.
   * @param knn_indices nearest neighbor indices (size m*n_neighbors)
   * @param knn_dists nearest neighbor distances (size m*n_neighbors
   * @param m number of samples in X
   * @param n_neighbors the number of neighbors to query for knn graph construction
   * @param n_components the number of components to project the X into
   * @param out output array for labels (size m)
   */
void fit_embedding(const cumlHandle &handle, long *knn_indices,
                   float *knn_dists, int m, int n_neighbors, int n_components,
                   float *out) {
  const auto &impl = handle.getImpl();
  MLCommon::Spectral::fit_embedding(
    impl.getcusparseHandle(), knn_indices, knn_dists, m, n_neighbors,
    n_components, out, handle.getDeviceAllocator(), handle.getStream());
}

/***
   * Given a feature matrix, this function computes the spectral
   * embeddings (lowest n_components eigenvectors), using
   * Lanczos min cut algorithm.
   * @param X a feature matrix (size m*n)
   * @param m number of samples in X
   * @param n number of features in X
   * @param n_neighbors the number of neighbors to query for knn graph construction
   * @param n_components the number of components to project the X into
   * @param out output array for labels (size m)
   */
void fit_embedding(const cumlHandle &handle, float *X, int m, int n,
                   int n_neighbors, int n_components, float *out) {
  const auto &impl = handle.getImpl();
  MLCommon::Spectral::fit_embedding(
    impl.getcusparseHandle(), X, m, n, n_neighbors, n_components, out,
    handle.getDeviceAllocator(), handle.getStream());
}

}  // namespace Spectral
}  // namespace ML
