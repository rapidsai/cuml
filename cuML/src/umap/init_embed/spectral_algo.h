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

#include "umap/umapparams.h"

#include <nvgraph.h>
#include <iostream>

#pragma once

namespace UMAPAlgo {

    namespace InitEmbed {

        namespace SpectralInit {

            using namespace ML;

            void check(nvgraphStatus_t status) {
                if (status != NVGRAPH_STATUS_SUCCESS) {
                    printf("ERROR : %d\n",status);
                    exit(0);
                }
            }


            template<typename T>
            void launcher(const T *X, int n, int d,
                          const long *knn_indices, const T *knn_dists,
                          int *rows, int *cols, float *vals,
                          int nnz,
                          UMAPParams *params,
                          T *embedding) {

                nvgraphHandle_t handle;
                cudaDataType_t edge_dimT = CUDA_R_32F;
                check(nvgraphCreate (&handle));

                std::cout << "Running spectral initializer" << std::endl;

                /**
                 * First convert COO to CSR
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

                std::cout << "converting to csr" << std::endl;

                nvgraphCSRTopology32I_st *CSR_input = new nvgraphCSRTopology32I_st();
                CSR_input->destination_indices = dst_indices;
                CSR_input->nedges = nnz;
                CSR_input->nvertices = n;
                CSR_input->source_offsets = src_offsets;

                check(nvgraphConvertTopology(handle, NVGRAPH_COO_32, (void*)COO_input, (void*)vals,
                        &edge_dimT, NVGRAPH_CSR_32, (void*)CSR_input, (void*)vals));

                std::cout << "done." << std::endl;

                /**
                 * Calculate the eigenvectors (ordered by eigenvalue)
                 * of the normalized laplacian from the 1-skeleton
                 */

                int weight_index = 0;
                int *clustering;
                MLCommon::allocate(clustering, n);

                float *eigVals;
                MLCommon::allocate(eigVals, params->n_components);

                // Spectral clustering parameters
                struct SpectralClusteringParameter clustering_params;
                clustering_params.n_clusters = params->n_components;
                clustering_params.n_eig_vects = params->n_components;
                clustering_params.algorithm = NVGRAPH_BALANCED_CUT_LANCZOS;
                clustering_params.evs_tolerance = 0.0f;
                clustering_params.evs_max_iter = 0;
                clustering_params.kmean_tolerance = 0.0f;
                clustering_params.kmean_max_iter = 1;

                nvgraphGraphDescr_t graph;
                check(nvgraphCreateGraphDescr(handle, &graph));
                check(nvgraphSetGraphStructure(handle, graph, (void*)CSR_input, NVGRAPH_CSR_32));
                check(nvgraphAllocateEdgeData(handle, graph, 1, &edge_dimT));
                check(nvgraphSetEdgeData(handle, graph, (void*)vals, 0));

                check(nvgraphSpectralClustering(handle, graph, weight_index, &clustering_params, clustering, eigVals, embedding));

                check(nvgraphDestroyGraphDescr(handle, graph));
                check(nvgraphDestroy(handle));

                CUDA_CHECK(cudaFree(src_offsets));
                CUDA_CHECK(cudaFree(dst_indices));

                free(COO_input);
                free(CSR_input);

                CUDA_CHECK(cudaFree(clustering));
                CUDA_CHECK(cudaFree(eigVals));

                std::cout << "Spectral embedding complete" << std::endl;
            }
        }
    }
};
