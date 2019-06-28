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

#include <linalg/eltwise.h>
#include <float.h>
#include <math.h>
#include "utils.h"

namespace ML {
namespace TSNE {

/****************************************/
/* Finds the best guassian bandwith for
    each row in the dataset             */
__global__ void 
sigmas_kernel(const float *__restrict__ distances,
             float *__restrict__ P,
             const float perplexity,
             const float desired_entropy,
             float *__restrict__ P_sum,
             const int epochs, const float tol,
             const int n, const int k)
{
    // For every item in row
    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= n) return;

    float beta_min = -INFINITY, beta_max = INFINITY;
    float beta = 1;
    float sum_P_row = 0;
    register const int ik = i*k;

    for (int step = 0; step < epochs; step++) {
        float sum_Pi = FLT_EPSILON;

        // Exponentiate to get guassian
        for (int j = 0; j < k; j++) {
            P[ik + j] = __expf(-distances[ik + j] * beta);
            sum_Pi += P[ik + j];
        }

        // Normalize
        float sum_disti_Pi = 0;
        sum_P_row = 0;
        const float div = __fdividef(1.0f, sum_Pi);
        for (int j = 0; j < k; j++) {
            P[ik + j] *= div;
            sum_disti_Pi += distances[ik + j] * P[ik + j];
            sum_P_row += P[ik + j];
        }

        const float entropy = __logf(sum_Pi) + beta * sum_disti_Pi;
        const float entropy_diff = entropy - desired_entropy;
        if (fabs(entropy_diff) <= tol) 
            break;

        // Bisection search
        if (entropy_diff > 0) {
            beta_min = beta;
            if (isinf(beta_max))
                beta *= 2.0f;
            else
                beta = (beta + beta_max) * 0.5f;
        }
        else {
            beta_max = beta;
            if (isinf(beta_min))
                beta *= 0.5f;
            else
                beta = (beta + beta_min) * 0.5f;
        }
    }
    P_sum[i] = sum_P_row;
}


/****************************************/
float
perplexity_search(const float *__restrict__ distances,
                 float *__restrict__ P, const float perplexity,
                 const int epochs, const float tol, const int n,
                 const int dim, cudaStream_t stream)
{
    const float desired_entropy = logf(perplexity);
    float *P_sum; CUDA_CHECK(cudaMalloc(&P_sum, sizeof(float)*n));

    sigmas_kernel<<<ceil(n, 1024), 1024, 0, stream>>>(
        distances, P, perplexity, desired_entropy, P_sum, epochs, tol, n, dim);
    CUDA_CHECK(cudaPeekAtLastError());

    thrust::device_ptr<float> begin = thrust::device_pointer_cast(P_sum);
    float sum = thrust::reduce(thrust::cuda::par.on(stream), begin, begin + n);
    CUDA_CHECK(cudaFree(P_sum));

    return sum;
}


/****************************************/
/* Compute attractive forces in O(uN) time.
    Uses only nearest neighbors         */
__global__ void
attractive_kernel(const float *__restrict__ VAL,
                const int *__restrict__ COL,
                const int *__restrict__ ROW,
                const float *__restrict__ Y,
                const float *__restrict__ norm,
                float *__restrict__ attract,
                const int NNZ, const int n, const int dim,
                const float df_power, // -(df + 1)/2)
                const float recp_df)  // 1 / df
{
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= NNZ) return;
    const int i = ROW[index], j = COL[index];

    // Euclidean distances
    // TODO: can provide any distance ie cosine
    float d = 0;
    for (int k = 0; k < dim; k++)
        d += Y[k*n + i] * Y[k*n + j];
    const float euclidean_d = -2.0f*d + norm[i] + norm[j];

    // TODO: Calculate Kullback-Leibler divergence
    const float PQ = VAL[index] *  __powf((1.0f + euclidean_d*recp_df), df_power); // P*Q

    // Apply forces
    for (int k = 0; k < dim; k++)
        atomicAdd(&attract[k*n + i] ,   PQ * (Y[k*n + i] - Y[k*n + j])  );
}


/****************************************/
/* Special case when dim == 2. Can speed
    up many calculations up             */
__global__ void
attractive_kernel_2d(const float *__restrict__ VAL,
                    const int *__restrict__ COL,
                    const int *__restrict__ ROW,
                    const float *__restrict__ Y1,
                    const float *__restrict__ Y2,
                    const float *__restrict__ norm,
                    float *__restrict__ attract1,
                    float *__restrict__ attract2,
                    const int NNZ)
{
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= NNZ) return;
    const int i = ROW[index], j = COL[index];

    // Euclidean distances
    // TODO: can provide any distance ie cosine
    const float euclidean_d = norm[i] + norm[j] - 2.0f*(Y1[i]*Y1[j] + Y2[i]*Y2[j]);

    // TODO: Calculate Kullback-Leibler divergence
    const float PQ =  __fdividef(VAL[index] , (1.0f + euclidean_d)); // P*Q
    
    // Apply forces
    atomicAdd(&attract1[i] ,  PQ * (Y1[i] - Y1[j])  );
    atomicAdd(&attract2[i] ,  PQ * (Y2[i] - Y2[j])  );
}


/****************************************/
void
attractive_forces(const float *__restrict__ VAL,
                const int *__restrict__ COL,
                const int *__restrict__ ROW,
                const float *__restrict__ Y,
                const float *__restrict__ norm,
                float *__restrict__ attract,
                const int NNZ, const int n, const int dim,
                const float df_power, // -(df + 1)/2)
                const float recp_df,  // 1 / df
                cudaStream_t stream)
{
    CUDA_CHECK(cudaMemset(attract, 0, sizeof(float)*n*dim));

    // TODO: Calculate Kullback-Leibler divergence
    // For general embedding dimensions
    if (dim != 2) {
        attractive_kernel<<<ceil(NNZ, 1024), 1024, 0, stream>>>(
            VAL, COL, ROW, Y, norm, attract, NNZ, n, dim, df_power, recp_df);
    }
    // For special case dim == 2
    else {
        attractive_kernel_2d<<<ceil(NNZ, 1024), 1024, 0, stream>>>(
            VAL, COL, ROW, Y, Y + n, norm, attract, attract + n, NNZ);
    }
    CUDA_CHECK(cudaPeekAtLastError());
}


/****************************************/
/* Computes repulsive forces in pseudo-O(N^2)
    time where many of the math ops are
    made considerably faster.           */
__global__ void
repulsive_kernel(const float *__restrict__ Y,
                float *__restrict__ repel,
                const float *__restrict__ norm,
                float *__restrict__ Z_sum1,
                float *__restrict__ Z_sum2,
                const int n, const int dim,
                const float df_power, // -(df + 1)/2)
                const float recp_df)  // 1 / df
{
    const int j = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every item in row
    const int i = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every row
    if (j >= i || i >= n || j >= n) return;

    // Euclidean distances
    // TODO: can provide any distance ie cosine
    float d = 0;
    for (int k = 0; k < dim; k++)
        d += Y[k*n + i] * Y[k*n + j];
    const float euclidean_d = -2.0f*d + norm[i] + norm[j];

    // Q and Q^2
    const float Q = __powf((1.0f + euclidean_d*recp_df)  ,  df_power);
    const float Q2 = Q*Q;

    // Apply forces
    for (int k = 0; k < dim; k++) {
        const float force = Q2 * (Y[k*n + j] - Y[k*n + i]);
        atomicAdd(&repel[k*n + i] , force);
        atomicAdd(&repel[k*n + j] , force);
    }

    // Sum up Z sum
    if (i % 2 == 0)
        atomicAdd(&Z_sum1[i] , Q);
    else
        atomicAdd(&Z_sum2[i] , Q);
}


/****************************************/
/* Special case when dim == 2. Much faster
    since calculations are streamlined. */
__global__ void
repulsive_kernel_2d(const float *__restrict__ Y1,
                    const float *__restrict__ Y2,
                    float *__restrict__ repel1,
                    float *__restrict__ repel2,
                    const float *__restrict__ norm,
                    float *__restrict__ Z_sum1,
                    float *__restrict__ Z_sum2,
                    const int n)
{
    const int j = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every item in row
    const int i = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every row
    if (j >= i || i >= n || j >= n) return;

    // Euclidean distances
    // TODO: can provide any distance ie cosine
    const float euclidean_d = norm[i] + norm[j] - 2.0f*(Y1[i]*Y1[j] + Y2[i]*Y2[j]);
    const float Q = __fdividef(1.0f , (1.0f + euclidean_d));
    const float Q2 = Q*Q;

    const float force1 = Q2 * (Y1[j] - Y1[i]);
    const float force2 = Q2 * (Y2[j] - Y2[i]);

    // Add forces
    atomicAdd(&repel1[i] , force1);
    atomicAdd(&repel1[j] , - force1);
    
    atomicAdd(&repel2[i] , force2);
    atomicAdd(&repel2[j] , - force2);

    // Sum up Z sum
    if (i % 2 == 0)
        atomicAdd(&Z_sum1[i] , Q);
    else
        atomicAdd(&Z_sum2[i] , Q);
}


/****************************************/
template <int TPB_X = 32, int TPB_Y = 32> float
repulsive_forces(const float *__restrict__ Y,
                float *__restrict__ repel,
                const float *__restrict__ norm,
                float *__restrict__ Z_sum,
                const int n, const int dim,
                const float df_power, // -(df + 1)/2)
                const float recp_df,
                cudaStream_t stream)
{
    CUDA_CHECK(cudaMemset(Z_sum, 0, sizeof(float) * 2 * n));
    CUDA_CHECK(cudaMemset(repel, 0, sizeof(float) * n * dim));

    const dim3 threadsPerBlock(TPB_X, TPB_Y);
    const dim3 numBlocks(ceil(n, threadsPerBlock.x), ceil(n, threadsPerBlock.y));

    // For general embedding dimensions
    if (dim != 2) {
        repulsive_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
            Y, repel, norm, Z_sum, Z_sum + n, n, dim, df_power, recp_df);
    }
    // For special dim == 2 case
    else {
        repulsive_kernel_2d<<<numBlocks, threadsPerBlock, 0, stream>>>(
            Y, Y + n, repel, repel + n, norm, Z_sum, Z_sum + n, n);
    }
    CUDA_CHECK(cudaPeekAtLastError());

    // Find sum(Z_sum)
    thrust::device_ptr<float> begin = thrust::device_pointer_cast(Z_sum);
    float Z = thrust::reduce(thrust::cuda::par.on(stream), begin, begin + 2 * n);
    return 1.0f / (2.0f * (Z + (float)n)); // Notice + n since diagonal of repulsion sums to n
}


/****************************************/
/* Applys or integrates all forces. Uses
    more gains and contrains the output
    for output stability                */
__global__ void
apply_kernel(float *__restrict__ Y,
            float *__restrict__ velocity,
            const float *__restrict__ attract,
            const float *__restrict__ repel,
            float *__restrict__ means,
            float *__restrict__ gains,
            const float Z, // sum(Q)
            const float learning_rate,
            const float C, // constant from T-Dist Degrees of Freedom
            const float momentum,
            const float SIZE,  // SIZE = n*dim
            const int n, const float min_gain,
            float *__restrict__ gradient,
            const bool check_convergence)
{
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= SIZE) return;

    const float dy = C * (attract[index] + Z * repel[index]);
    if (check_convergence)
        gradient[index] = dy*dy;

    // Find new gain
    // TODO: Incorporate AadaBound (2019) or Adam
    if (signbit(dy) != signbit(velocity[index]))
        gains[index] += 0.2f; // Original TSNE is 0.2
    else
        gains[index] *= 0.8f; // Original TSNE is 0.8
    if (gains[index] < min_gain)
        gains[index] = min_gain;

    velocity[index] = momentum * velocity[index] - learning_rate * dy * gains[index];
    Y[index] += velocity[index];

    // Add to mean
    //atomicAdd(&means[index / n], Y[index]);
}


/****************************************/
template <int TPB_X = 32, int TPB_Y = 32> float
apply_forces(float *__restrict__ Y,
            float *__restrict__ velocity,
            const float *__restrict__ attract,
            const float *__restrict__ repel,
            float *__restrict__ means,
            float *__restrict__ gains,
            const float Z, // sum(Q)
            const float learning_rate,
            const float C, // constant from T-dist
            const float momentum,
            const float dim, const int n,
            const float min_gain,
            float *__restrict__ gradient,
            const bool check_convergence,
            cudaStream_t stream)
{
    //cudaMemset(means, 0, sizeof(float) * dim);
    if (check_convergence)
        CUDA_CHECK(cudaMemset(gradient, 0, sizeof(float) * n*dim));

    apply_kernel<<<ceil(n*dim, 1024), 1024, 0, stream>>>(Y, velocity,
        attract, repel, means, gains, Z, learning_rate, C, momentum,
        n*dim, n, min_gain, gradient, check_convergence);
    CUDA_CHECK(cudaPeekAtLastError());


    // Find sum of gradient norms
    float gradient_norm = INFINITY;
    if (check_convergence) {
        thrust::device_ptr<float> begin = thrust::device_pointer_cast(gradient);
        gradient_norm = sqrtf(thrust::reduce(thrust::cuda::par.on(stream), begin, begin + n*dim));
    }

    // TODO: Subtract means
    return gradient_norm;
}


}
}
