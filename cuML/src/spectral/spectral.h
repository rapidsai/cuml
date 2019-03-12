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

#pragma once

#include <nvgraph.h>

#include "sparse/nvgraph_wrappers.h"

#include "sparse/coo.h"
#include "knn/knn.h"

#include "cuda_utils.h"

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

        template<typename T>
        void fit_clusters(int *rows, int *cols, T *vals, int nnz,
                int n, int n_clusters, float eigen_tol, int *out) {

            nvgraphHandle_t handle;
            cudaDataType_t edge_dimT = CUDA_R_32F;
            NVGRAPH_CHECK(nvgraphCreate (&handle));

            /**
             * Convert COO to CSR
             *
             * todo: Add this to sparse prims
             */

            // Allocate csr arrays
            int *src_offsets, *dst_indices;
            MLCommon::allocate(src_offsets, n+1);
            MLCommon::allocate(dst_indices, nnz);

            nvgraphCOOTopology32I_st *COO_input = new nvgraphCOOTopology32I_st();
            COO_input->nedges = nnz;
            COO_input->nvertices = n;
            COO_input->source_indices = rows;
            COO_input->destination_indices = cols;

            nvgraphCSRTopology32I_st *CSR_input = new nvgraphCSRTopology32I_st();
            CSR_input->destination_indices = dst_indices;
            CSR_input->nedges = nnz;
            CSR_input->nvertices = n;
            CSR_input->source_offsets = src_offsets;

            NVGRAPH_CHECK(nvgraphConvertTopology(handle,
                    NVGRAPH_COO_32, (void*)COO_input, (void*)vals,
                    &edge_dimT, NVGRAPH_CSR_32, (void*)CSR_input, (void*)vals));

            int weight_index = 0;

            float *eigVals, *embedding;
            MLCommon::allocate(eigVals, n_clusters);
            MLCommon::allocate(embedding, n*n_clusters);

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
            NVGRAPH_CHECK(nvgraphCreateGraphDescr(handle, &graph));
            NVGRAPH_CHECK(nvgraphSetGraphStructure(handle, graph,
                    (void*)CSR_input, NVGRAPH_CSR_32));
            NVGRAPH_CHECK(nvgraphAllocateEdgeData(handle, graph, 1, &edge_dimT));
            NVGRAPH_CHECK(nvgraphSetEdgeData(handle, graph, (void*)vals, 0));

            NVGRAPH_CHECK(nvgraphSpectralClustering(handle, graph, weight_index,
                    &clustering_params, out, eigVals, embedding));

            NVGRAPH_CHECK(nvgraphDestroyGraphDescr(handle, graph));
            NVGRAPH_CHECK(nvgraphDestroy(handle));

            CUDA_CHECK(cudaFree(src_offsets));
            CUDA_CHECK(cudaFree(dst_indices));
            CUDA_CHECK(cudaFree(embedding));
            CUDA_CHECK(cudaFree(eigVals));

            free(COO_input);
            free(CSR_input);
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
        template<typename T>
        void fit_clusters(long *knn_indices, T *knn_dists, int m, int n_neighbors,
                int n_clusters, float eigen_tol, int *out) {

            int *rows, *cols;
            T *vals;

            MLCommon::allocate(rows, m*n_neighbors);
            MLCommon::allocate(cols, m*n_neighbors);
            MLCommon::allocate(vals, m*n_neighbors);

            MLCommon::Sparse::from_knn(knn_indices, knn_dists, m, n_neighbors,
                    rows, cols, vals);

            // todo: Need to symmetrize the knn to create the knn graph

            fit_clusters(rows, cols, vals, m*n_neighbors, m, n_clusters, eigen_tol, out);

            CUDA_CHECK(cudaFree(rows));
            CUDA_CHECK(cudaFree(cols));
            CUDA_CHECK(cudaFree(vals));
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
        template<typename T>
        void fit_clusters(T *X, int m, int n, int n_neighbors,
                int n_clusters, float eigen_tol, int *out) {

            kNN *knn = new kNN(n);

            long *knn_indices;
            float *knn_dists;

            MLCommon::allocate(knn_indices, m*n_neighbors);
            MLCommon::allocate(knn_dists, m*n_neighbors);

            kNNParams params[1];
            params[0].N = m;
            params[0].ptr = X;


            knn->fit(*&params, 1);
            knn->search(X, m, knn_indices, knn_dists, n_neighbors);

            fit_clusters(knn_indices, knn_dists, m, n_neighbors,
                    n_clusters, eigen_tol, out);

            CUDA_CHECK(cudaFree(knn_indices));
            CUDA_CHECK(cudaFree(knn_dists));

            delete knn;
        }

        /***
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
         * @param out output array for labels (size m)
         */
        template<typename T>
        void fit_embedding(int *rows, int*cols, T *vals, int nnz, int n,
                int n_components, T *out) {

            nvgraphHandle_t handle;
            cudaDataType_t edge_dimT = CUDA_R_32F;
            NVGRAPH_CHECK(nvgraphCreate (&handle));

            // Allocate csr arrays
            int *src_offsets, *dst_indices;
            MLCommon::allocate(src_offsets, n+1);
            MLCommon::allocate(dst_indices, nnz);

            nvgraphCOOTopology32I_st *COO_input = new nvgraphCOOTopology32I_st();
            COO_input->nedges = nnz;
            COO_input->nvertices = n;
            COO_input->source_indices = rows;
            COO_input->destination_indices = cols;

            nvgraphCSRTopology32I_st *CSR_input = new nvgraphCSRTopology32I_st();
            CSR_input->destination_indices = dst_indices;
            CSR_input->nedges = nnz;
            CSR_input->nvertices = n;
            CSR_input->source_offsets = src_offsets;

            NVGRAPH_CHECK(nvgraphConvertTopology(handle, NVGRAPH_COO_32,
                    (void*)COO_input, (void*)vals,
                    &edge_dimT, NVGRAPH_CSR_32, (void*)CSR_input, (void*)vals));

            int weight_index = 0;

            float *eigVals;
            int *labels;
            MLCommon::allocate(labels, n);
            MLCommon::allocate(eigVals, n_components);

            // Spectral clustering parameters
            struct SpectralClusteringParameter clustering_params;
            clustering_params.n_clusters = n_components;
            clustering_params.n_eig_vects = n_components;
            clustering_params.algorithm = NVGRAPH_BALANCED_CUT_LANCZOS;
            clustering_params.evs_tolerance = 0.0f;
            clustering_params.evs_max_iter = 0;
            clustering_params.kmean_tolerance = 0.0f;
            clustering_params.kmean_max_iter = 1;

            nvgraphGraphDescr_t graph;
            NVGRAPH_CHECK(nvgraphCreateGraphDescr(handle, &graph));
            NVGRAPH_CHECK(nvgraphSetGraphStructure(handle, graph,
                    (void*)CSR_input, NVGRAPH_CSR_32));
            NVGRAPH_CHECK(nvgraphAllocateEdgeData(handle, graph, 1, &edge_dimT));
            NVGRAPH_CHECK(nvgraphSetEdgeData(handle, graph, (void*)vals, 0));

            NVGRAPH_CHECK(nvgraphSpectralClustering(handle, graph, weight_index,
                    &clustering_params, labels, eigVals, out));

            NVGRAPH_CHECK(nvgraphDestroyGraphDescr(handle, graph));
            NVGRAPH_CHECK(nvgraphDestroy(handle));

            CUDA_CHECK(cudaFree(src_offsets));
            CUDA_CHECK(cudaFree(dst_indices));
            CUDA_CHECK(cudaFree(eigVals));
            CUDA_CHECK(cudaFree(labels));

            free(COO_input);
            free(CSR_input);
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
        template<typename T>
        void fit_embedding(long *knn_indices, float *knn_dists, int m, int n_neighbors,
            int n_components, T *out) {

            int *rows, *cols;
            T *vals;

            MLCommon::allocate(rows, m*n_neighbors);
            MLCommon::allocate(cols, m*n_neighbors);
            MLCommon::allocate(vals, m*n_neighbors);

            MLCommon::Sparse::from_knn(knn_indices, knn_dists, m, n_neighbors,
                    rows, cols, vals);

            // todo: Need to symmetrize the knn graph here. UMAP works here because
            // it has already done this.

            fit_embedding(rows, cols, vals, m*n_neighbors, m, n_components, out);

            CUDA_CHECK(cudaFree(rows));
            CUDA_CHECK(cudaFree(cols));
            CUDA_CHECK(cudaFree(vals));
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
        template<typename T>
        void fit_embedding(T *X, int m, int n,
                int n_neighbors, int n_components,
                T *out) {

            kNN *knn = new kNN(n);

            long *knn_indices;
            float *knn_dists;

            MLCommon::allocate(knn_indices, m*n_neighbors);
            MLCommon::allocate(knn_dists, m*n_neighbors);

            kNNParams params[1];
            params[0].N = m;
            params[0].ptr = X;

            knn->fit(*&params, 1);
            knn->search(X, m, knn_indices, knn_dists, n_neighbors);

            fit_embedding(knn_indices, knn_dists, m, n_neighbors,
                    n_components, out);

            CUDA_CHECK(cudaFree(knn_indices));
            CUDA_CHECK(cudaFree(knn_dists));

            delete knn;
        }
    }
}
