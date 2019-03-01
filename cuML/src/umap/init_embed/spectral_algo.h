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
#include <cusparse_v2.h>

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
                          UMAPParams *params,
                          T *embedding) {

                /**
                 * Calculate the eigenvectors (ordered by eigenvalue)
                 * of the normalized laplacian from the 1-skeleton
                 */

                float *csrValA_h = (float*)malloc(sizeof(float));
                int *csrRowPtrA_h = (int*)malloc(sizeof(int));
                int *csrColIndA_h = (int*)malloc(sizeof(int));

                const size_t nnz = 1, n_ev = params->n_components, edge_numsets = 1;

                int weight_index = 0, *clustering_h;
                float *eigVals_h, *eigVecs_h;

                nvgraphHandle_t handle;
                nvgraphGraphDescr_t graph;
                cudaDataType_t edge_dimT = CUDA_R_32F;
                nvgraphCSRTopology32I_st CSR_input = {n, nnz, csrRowPtrA_h, csrColIndA_h};

                // Allocate host data for nvgraphSpectralClustering output
                clustering_h = (int*)malloc(n*sizeof(int));
                eigVals_h = (float*)malloc(n_ev*sizeof(float));
                eigVecs_h = (float*)malloc(n_ev*n*sizeof(float));

                // Spectral clustering parameters
                struct SpectralClusteringParameter clustering_params;
                clustering_params.n_clusters = n_ev;
                clustering_params.n_eig_vects = n_ev;
                clustering_params.algorithm = NVGRAPH_MODULARITY_MAXIMIZATION;
                clustering_params.evs_tolerance = 0.0f;
                clustering_params.evs_max_iter = 0;
                clustering_params.kmean_tolerance = 0.0f;
                clustering_params.kmean_max_iter = 0;

                check(nvgraphCreate (&handle));
                check(nvgraphCreateGraphDescr(handle, &graph));
                check(nvgraphSetGraphStructure(handle, graph, (void*)&CSR_input, NVGRAPH_CSR_32));
                check(nvgraphAllocateEdgeData(handle, graph, edge_numsets, &edge_dimT));
                check(nvgraphSetEdgeData(handle, graph, (void*)csrValA_h, 0));

                check(nvgraphSpectralClustering(handle, graph, weight_index, &clustering_params, clustering_h, eigVals_h, eigVecs_h));

            }
        }
    }
};
