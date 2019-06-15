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

#include "common/cumlHandle.hpp"

#include "tsne/tsne.h"

#include "cublas_v2.h"
#include "distances.h"
#include "slow_kernels.h"
#include "fast_kernels.h"
#include "utils.h"
#define cuda_max_potential cudaOccupancyMaxPotentialBlockSize


namespace ML {
using namespace MLCommon;

void TSNE_fit(const cumlHandle &handle, const float *X, float *Y, const int n,
            const int p, const int n_components = 2, int n_neighbors = 30,
            const float perplexity = 30.0f, const int perplexity_max_iter = 100,
            const int perplexity_tol = 1e-5,
            const float early_exaggeration = 12.0f,
            const int exaggeration_iter = 150, const float min_gain = 0.01f,
            const float eta = 500.0f, const int max_iter = 500,
            const float pre_momentum = 0.8, const float post_momentum = 0.5,
            const long long seed = -1, const bool initialize_embeddings = true,
            const bool verbose = true, const int method = 1)
    // Method = 0 for Naive, 1 for Fast
{
    assert(n > 0 && p > 0 && n_components > 0 && n_neighbors > 0 && X != NULL && Y != NULL);
    if (verbose)
        printf("[Info]  Data size = (%d, %d) with n_components = %d\n", n, p, n_components);

    auto d_alloc = handle.getDeviceAllocator();
    cudaStream_t stream = handle.getStream();

    if (n_neighbors > n) n_neighbors = n;

    // Some preliminary intializations for cuBLAS and cuML
    const int k = n_components;
    cublasHandle_t BLAS = handle.getImpl().getCublasHandle();


    // Get distances
    if (verbose) printf("[Info] Getting distances.\n");
    float *distances = (float *)d_alloc->allocate(sizeof(float) * n * n_neighbors, stream);
    long *indices = (long *)d_alloc->allocate(sizeof(long) * n * n_neighbors, stream);

    get_distances(X, n, p, indices, distances, n_neighbors, stream);


    // Normalize distances
    if (verbose) printf("[Info] Now normalizing distances so exp(D) doesn't explode.\n");
    normalize_distances(n, distances, n_neighbors, stream);


    // Optimal perplexity
    if (verbose) printf("[Info] Searching for optimal perplexity via bisection search.\n");
    float *P = (float *)d_alloc->allocate(sizeof(float) * n * n_neighbors, stream);

    // Determine best blocksize / gridsize
    int blockSize_N = 1024; int minGridSize_N;
    cuda_max_potential(&minGridSize_N, &blockSize_N, __determine_sigmas, 0, n);
    const int gridSize_N = ceil(n, blockSize_N);


    const float P_sum = determine_sigmas(distances, P, perplexity, perplexity_max_iter,
                                        perplexity_tol, n, n_neighbors, stream, gridSize_N, blockSize_N);
    d_alloc->deallocate(distances, sizeof(float) * n * n_neighbors, stream);
    if (verbose) printf("[Info] Perplexity sum = %f\n", P_sum);


    // Convert data to COO layout
    COO_t<float> P_PT;
    symmetrize_perplexity(P, indices, &P_PT, n, n_neighbors, P_sum, early_exaggeration, stream, handle);
        
    const int NNZ = P_PT.nnz;
    float *VAL = P_PT.vals;
    const int *COL = P_PT.rows;
    const int *ROW = P_PT.cols;


    // Allocate data [NOTICE Fortran Contiguous for method = Naive and C-Contiguous for fast]
    if (initialize_embeddings)
        random_vector(Y, -0.03f, 0.03f, n * k, stream, seed);


    // Allocate space
    if (verbose) printf("[Info] Now allocating memory for TSNE.\n");
    float *norm = (float *)d_alloc->allocate(sizeof(float) * n, stream);
    float *Q_sum = (float *)d_alloc->allocate(sizeof(float) * n, stream);
    double *sum = (double *)d_alloc->allocate(sizeof(double), stream);

    float *attract = (float *)d_alloc->allocate(sizeof(float) * n * k, stream);
    float *repel = (float *)d_alloc->allocate(sizeof(float) * n * k, stream);

    float *iY = (float *)d_alloc->allocate(sizeof(float) * n * k, stream);
    float *gains = (float *)d_alloc->allocate(sizeof(float) * n * k, stream);
    float *means = (float*)d_alloc->allocate(sizeof(float) * k, stream);


    // Compute optimal gridSize and blockSize for attractive forces
    int blockSize_NNZ = 1024; int minGridSize_NNZ;
    if (n_components == 2)
        cuda_max_potential(&minGridSize_NNZ, &blockSize_NNZ, __attractive_fast_2dim, 0, NNZ);
    else
        cuda_max_potential(&minGridSize_NNZ, &blockSize_NNZ, __attractive_fast, 0, NNZ);
    const int gridSize_NNZ = ceil(NNZ, blockSize_NNZ);

    // Compute optimal gridSize and blockSize for applying forces
    int blockSize_dimN = 1024; int minGridSize_dimN;
    cuda_max_potential(&minGridSize_dimN, &blockSize_dimN, __apply_forces, 0, n_components*n);
    const int gridSize_dimN = ceil(n_components*n, blockSize_dimN);


    // Do gradient updates
    float momentum = pre_momentum;
    float Z;

    if (verbose) printf("[Info] Start gradient updates!\n");
    if (method == 1) {
        /*
        Fast algorithm uses 0 matrices, and does all computations within
        the GPU registers. This guarantees no memory movement and so can
        be fast in practice.
        */
        for (int iter = 0; iter < max_iter; iter++) {
            if (iter == exaggeration_iter) {
                momentum = post_momentum;
                // Divide perplexities
                const float div = 1.0f / early_exaggeration;
                array_multiply(VAL, NNZ, div, stream);
            }
            // Get norm(Y)
            get_norm_fast(Y, norm, n, k, stream, gridSize_N, blockSize_N);
            
            // Fast compute attractive forces from COO matrix
            attractive_fast(VAL, COL, ROW, Y, norm, attract, NNZ, n, n_components, stream,
                gridSize_NNZ, blockSize_NNZ);

            // Fast compute repulsive forces
            Z = repulsive_fast(Y, repel, norm, Q_sum, n, n_components, stream);
            if (verbose && iter % 100 == 0)
                printf("[Info]  Z at iter = %d is %lf.\n", iter, Z);

            // Integrate forces with momentum
            apply_forces(attract, means, repel, Y, iY, gains, n, k, Z, min_gain, momentum, eta, stream,
                gridSize_dimN, blockSize_dimN);
        }
    }

    else if (method == 0) {
        int error;
        /*
        Naive algorithm uses cuBLAS to compute the full Y @ Y.T matrix.
        Code flow follows closely to Maaten's original TSNE code.
        Notice Naive is relatively memory hungry - uses O(N^2).
        Fast uses close to no extra memory.
        */
        const float neg2 = -2.0f, beta = 0.0f, one = 1.0f;
        float *Q = (float *)d_alloc->allocate(sizeof(float) * n * n, stream);

        for (int iter = 0; iter < max_iter; iter++) {
            if (iter == exaggeration_iter) {
                momentum = post_momentum;
                // Divide perplexities
                const float div = 1.0f / early_exaggeration;
                array_multiply(VAL, NNZ, div, stream);
            }
            // Get norm(Y)
            get_norm_slow(Y, norm, n, k, stream);

            // Find Y @ Y.T
            error = cublasSsyrk(BLAS, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k,
                                                &neg2, Y, n, &beta, Q, n);
            if (error != 0) {
                if (verbose) printf("[ERROR]    BLAS failed. Terminating TSNE\n");
                break;
            }

            // Form T = 1 / (1 + d) = 1 / (1 + -2*Y@Y.T )
            Z = form_t_distribution(Q, norm, n, Q_sum, sum, stream);
            if (verbose && iter % 100 == 0)
                printf("[Info]  Z at iter = %d is %lf.\n", iter, Z);

            // Compute attractive forces from COO matrix
            attractive_forces(VAL, COL, ROW, Q, Y, attract, NNZ, n, k, stream);

            // Change Q to Q**2
            postprocess_Q(Q, Q_sum, n, stream);

            // Do Q**2 @ Y
            error = cublasSsymm(BLAS, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, n,
                                                k, &one, Q, n, Y, n, &beta, repel, n);
            if (error != 0) {
                if (verbose) printf("[ERROR]    BLAS failed. Terminating TSNE\n");
                break;
            }

            // Compute repel - Q**2 @ mean_Y
            repel_minus_QY(repel, Q_sum, Y, n, k, stream);

            // Integrate forces with momentum
            apply_forces_slow(attract, means, repel, Y, iY, gains, n, k, Z, min_gain, momentum, eta, stream);
        }

        d_alloc->deallocate(Q, sizeof(float) * n * n, stream);
    }

    P_PT.destroy();

    d_alloc->deallocate(norm, sizeof(float) * n, stream);
    d_alloc->deallocate(Q_sum, sizeof(float) * n, stream);
    d_alloc->deallocate(sum, sizeof(double), stream);

    d_alloc->deallocate(attract, sizeof(float) * n * k, stream);
    d_alloc->deallocate(repel, sizeof(float) * n * k, stream);

    d_alloc->deallocate(iY, sizeof(float) * n * k, stream);
    d_alloc->deallocate(gains, sizeof(float) * n * k, stream);
    d_alloc->deallocate(means, sizeof(float) * k, stream);
}


}  // namespace ML
