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

#include "trustworthiness.h"
#include <cuda_utils.h>
#include "distance/distance.h"
#include <selection/columnWiseSort.h>
#include "../knn/knn.h"

using namespace MLCommon;
using namespace MLCommon::Selection;
using namespace ML;

/**
* @brief Compute a kNN and returns the indexes of the nearest neighbors
* @input param input: Input matrix holding the dataset
* @input param n: Number of samples
* @input param d: Number of features
* @return Matrix holding the indexes of the nearest neighbors
*/
template<typename math_t>
long* get_knn(math_t* input, int n, int d, int n_neighbors)
{
    long* d_pred_I;
    math_t* d_pred_D;
    allocate<long>(d_pred_I, n*n_neighbors);
    allocate(d_pred_D, n*n_neighbors);

    kNNParams params = {input, n};
    kNN knn(d);
    knn.fit(&params, 1);
    knn.search(input, n, d_pred_I, d_pred_D, n_neighbors);

    long* h_pred_I = new long[n*n_neighbors];
    updateHost(h_pred_I, d_pred_I, n*n_neighbors);

    CUDA_CHECK(cudaFree(d_pred_I));
    CUDA_CHECK(cudaFree(d_pred_D));
    return h_pred_I;
}

namespace ML {

    /**
    * @brief Compute the trustworthiness score
    * @input param X: Data in original dimension
    * @input param X_embedded: Data in target dimension (embedding)
    * @input param n: Number of samples
    * @input param m: Number of features in high/original dimension
    * @input param d: Number of features in low/embedded dimension
    * @input param n_neighbors: Number of neighbors considered by trustworthiness score
    * @return Trustworthiness score
    */
    template<typename math_t>
    double cuml_trustworthiness(math_t* X, math_t* X_embedded, int n, int m, int d, int n_neighbors)
    {
        const int TMP_SIZE = MAX_BATCH_SIZE * n;

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        constexpr auto distance_type = MLCommon::Distance::DistanceType::EucUnexpandedL2Sqrt;
        size_t workspaceSize = 0; // EucUnexpandedL2Sqrt does not need any workspace
        typedef cutlass::Shape<8, 128, 128> OutputTile_t;
        bool bAllocWorkspace = false;

        math_t* d_pdist_tmp;
        allocate(d_pdist_tmp, TMP_SIZE);
        int* d_ind_X_tmp;
        allocate(d_ind_X_tmp, TMP_SIZE);
        int* h_ind_X = new int[n*n];

        int toDo = n;
        while (toDo > 0)
        {
            int batchSize = min(toDo, MAX_BATCH_SIZE); // Takes at most MAX_BATCH_SIZE vectors at a time

            MLCommon::Distance::distance<distance_type, math_t, math_t, math_t, OutputTile_t>
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

            updateHost(&h_ind_X[(n - toDo) * n], d_ind_X_tmp, batchSize * n, stream);

            toDo -= batchSize;
        }

        long* ind_X_embedded = get_knn(X_embedded, n, d, n_neighbors + 1);

        double t = 0.0;
        for (size_t i = 0; i < n; i++)
        {
            int* sample_i = &h_ind_X[i * n + 1];
            for (size_t j = 1; j <= n_neighbors; j++)
            {
                long idx = ind_X_embedded[i * (n_neighbors+1) + j];
                for (int r = 0; r < n-1; r++)
                {
                    if (sample_i[r] == idx)
                    {
                        t += max(0.0, double(r - n_neighbors));
                        break;
                    }
                }
            }
        }

        delete[] h_ind_X;
        delete[] ind_X_embedded;

        t = 1.0 - ((2.0 / ((n * n_neighbors) * ((2.0 * n) - (3.0 * n_neighbors) - 1.0))) * t);

        CUDA_CHECK(cudaStreamDestroy(stream));

        return t;
    }



    template double cuml_trustworthiness<float>(float* X, float* X_embedded, int n, int m, int d, int n_neighbors);
    //template double cuml_trustworthiness(double* X, double* X_embedded, int n, int m, int d, int n_neighbors);
    // Disabled for now as knn only takes floats

}