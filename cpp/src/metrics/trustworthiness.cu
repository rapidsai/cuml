/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#include "trustworthiness.h"
#include <cuda_utils.h>
#include "distance/distance.h"
#include <selection/columnWiseSort.h>
#include <common/cumlHandle.hpp>
#include <knn/knn.hpp>

using namespace MLCommon;
using namespace MLCommon::Distance;
using namespace MLCommon::Selection;
using namespace ML;

namespace ML {

    /**
    * @brief Compute a kNN and returns the indexes of the nearest neighbors
    * @input param input: Input matrix holding the dataset
    * @input param n: Number of samples
    * @input param d: Number of features
    * @return Matrix holding the indexes of the nearest neighbors
    */
    template<typename math_t>
    long* get_knn_indexes(const cumlHandle& h, math_t* input, int n,
                                int d, int n_neighbors)
    {
        cudaStream_t stream = h.getStream();
        auto d_alloc = h.getDeviceAllocator();
        
        long* d_pred_I = (long*)d_alloc->allocate(n * n_neighbors * sizeof(long), stream);
        math_t* d_pred_D = (math_t*)d_alloc->allocate(n * n_neighbors * sizeof(math_t), stream);

        ArrayPtr<float> params = {input, n};
        kNN knn(h, d);
        knn.fit(&params, 1);
        knn.search(input, n, d_pred_I, d_pred_D, n_neighbors);

        d_alloc->deallocate(d_pred_D, n * n_neighbors * sizeof(math_t), stream);
        return d_pred_I;
    }


    /**
    * @brief Compute a the rank of trustworthiness score
    * @input param ind_X: indexes given by pairwise distance and sorting
    * @input param ind_X_embedded: indexes given by KNN
    * @input param n: Number of samples
    * @input param n_neighbors: Number of neighbors considered by trustworthiness score
    * @input param work: Batch to consider (to do it at once use n * n_neighbors)
    * @output param rank: Resulting rank
    */
    template<typename math_t>
    __global__ void compute_rank(math_t *ind_X, long *ind_X_embedded,
                    int n, int n_neighbors, int work, double * rank)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= work)
            return;

        int n_idx = i / n_neighbors;
        int nn_idx = (i % n_neighbors) + 1;

        int idx = ind_X_embedded[n_idx * (n_neighbors+1) + nn_idx];
        math_t* sample_i = &ind_X[n_idx * n];
        for (int r = 1; r < n; r++)
        {
            if (sample_i[r] == idx)
            {
                int tmp = r - n_neighbors;
                if (tmp > 0)
                    atomicAdd(rank, tmp);
                break;
            }
        }
    }

    namespace Metrics {

        /**
        * @brief Compute the trustworthiness score
        * @input param X: Data in original dimension
        * @input param X_embedded: Data in target dimension (embedding)
        * @input param n: Number of samples
        * @input param m: Number of features in high/original dimension
        * @input param d: Number of features in low/embedded dimension
        * @input param n_neighbors: Number of neighbors considered by trustworthiness score
        * @input param distance_type: Distance type to consider
        * @return Trustworthiness score
        */
        template<typename math_t, DistanceType distance_type>
        double trustworthiness_score(const cumlHandle& h, math_t* X,
                            math_t* X_embedded, int n, int m, int d,
                            int n_neighbors)
        {
            const int TMP_SIZE = MAX_BATCH_SIZE * n;

            cudaStream_t stream = h.getStream();
            auto d_alloc = h.getDeviceAllocator();

            size_t workspaceSize = 0; // EucUnexpandedL2Sqrt does not reauire workspace (may need change for other distances)
            typedef cutlass::Shape<8, 128, 128> OutputTile_t;
            bool bAllocWorkspace = false;

            math_t* d_pdist_tmp = (math_t*)d_alloc->allocate(TMP_SIZE * sizeof(math_t), stream);
            int* d_ind_X_tmp = (int*)d_alloc->allocate(TMP_SIZE * sizeof(int), stream);

            long* ind_X_embedded = get_knn_indexes(h, X_embedded,
                n, d, n_neighbors + 1);

            double t_tmp = 0.0;
            double t = 0.0;
            double* d_t = (double*)d_alloc->allocate(sizeof(double), stream);

            int toDo = n;
            while (toDo > 0)
            {
                int batchSize = min(toDo, MAX_BATCH_SIZE);
                // Takes at most MAX_BATCH_SIZE vectors at a time

                distance<distance_type, math_t, math_t, math_t, OutputTile_t>
                        (&X[(n - toDo) * m], X,
                        d_pdist_tmp,
                        batchSize, n, m,
                        (void*)nullptr, workspaceSize,
                        stream
                );
                CUDA_CHECK(cudaPeekAtLastError());

                sortColumnsPerRow(d_pdist_tmp, d_ind_X_tmp,
                                    batchSize, n,
                                    bAllocWorkspace, NULL, workspaceSize,
                                    stream);
                CUDA_CHECK(cudaPeekAtLastError());

                t_tmp = 0.0;
                updateDevice(d_t, &t_tmp, 1, stream);

                int work = batchSize * n_neighbors;
                int n_blocks = work / N_THREADS + 1;
                compute_rank<<<n_blocks, N_THREADS, 0, stream>>>(d_ind_X_tmp,
                        &ind_X_embedded[(n - toDo) * (n_neighbors+1)],
                        n,
                        n_neighbors,
                        batchSize * n_neighbors,
                        d_t);
                CUDA_CHECK(cudaPeekAtLastError());

                updateHost(&t_tmp, d_t, 1, stream);
                t += t_tmp;

                toDo -= batchSize;
            }

            t = 1.0 - ((2.0 / ((n * n_neighbors) * ((2.0 * n) - (3.0 * n_neighbors) - 1.0))) * t);

            d_alloc->deallocate(ind_X_embedded, n * (n_neighbors + 1) * sizeof(long), stream);
            d_alloc->deallocate(d_pdist_tmp, TMP_SIZE * sizeof(math_t), stream);
            d_alloc->deallocate(d_ind_X_tmp, TMP_SIZE * sizeof(int), stream);
            d_alloc->deallocate(d_t, sizeof(double), stream);

            return t;
        }

        template double trustworthiness_score<float, EucUnexpandedL2Sqrt>(const cumlHandle& h,
            float* X, float* X_embedded, int n, int m, int d, int n_neighbors);

    }
}
