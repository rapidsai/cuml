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

#include <float.h>
#include <math.h>
#include "linalg/eltwise.h"
#include "linalg/matrix_vector_op.h"

#include <random/rng.h>
#include <stats/sum.h>
#include <sys/time.h>


namespace ML {
namespace TSNE {
using MLCommon::ceildiv;


void
random_vector(float *vector, const float minimum, const float maximum,
                     const int size, cudaStream_t stream, long long seed = -1) {
    if (seed <= 0) {
        // Get random seed based on time of day
        struct timeval tp;
        gettimeofday(&tp, NULL);
        seed = tp.tv_sec * 1000 + tp.tv_usec;
    }
    MLCommon::Random::Rng random(seed);
    random.uniform<float>(vector, size, minimum, maximum, stream);
    CUDA_CHECK(cudaPeekAtLastError());
}


__global__ void 
__determine_sigmas(const float *__restrict__ distances,
				 float *__restrict__ P,
				 const float perplexity,
				 const float desired_entropy,
				 float *__restrict__ P_sum,
				 const int epochs, const float tol,
				 const int n, const int k)
{
	// For every item in row
	const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < n) {
		float beta_min = -INFINITY;
		float beta_max = INFINITY;
		float beta = 1;
		float sum_Pi = 0;
		float sum_P_row, sum_disti_Pi;
		register const int ik = i * k;

		for (int step = 0; step < epochs; step++) {
			sum_Pi = 0;
			sum_disti_Pi = 0;
			sum_P_row = 0;

			// Exponentiate to get guassian
			for (int j = 0; j < k; j++) {
				P[ik + j] = __expf(-distances[ik + j] * beta);
				sum_Pi += P[ik + j];
			}
			if (sum_Pi == 0)
				sum_Pi = FLT_EPSILON;

			// Normalize
			// const float div_sum_Pi = 1.0f / sum_Pi;
			const float div_sum_Pi = __fdividef(1.0f, sum_Pi);
			for (int j = 0; j < k; j++) {
				P[ik + j] *= div_sum_Pi;  // P[i*k + j] / sum_Pi
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
		atomicAdd(P_sum, sum_P_row);
	}
}
float determine_sigmas(const float *__restrict__ distances,
					 float *__restrict__ P, const float perplexity,
					 const int epochs, const float tol, const int n,
					 const int k, cudaStream_t stream,
					 const int gridSize, const int blockSize,
					 const cumlHandle &handle) {
	const float desired_entropy = logf(perplexity);
	
	float *P_sum_ = (float*) handle.getDeviceAllocator()->allocate(sizeof(float), stream);
	float P_sum;
	CUDA_CHECK(cudaMemsetAsync(P_sum_, 0, sizeof(float), stream));

	__determine_sigmas<<<gridSize, blockSize, 0, stream>>>(
		distances, P, perplexity, desired_entropy, P_sum_, epochs, tol, n, k);
	CUDA_CHECK(cudaPeekAtLastError());

	// Store P_sum back into CPU
	MLCommon::updateHost<float>(&P_sum, P_sum_, 1, stream);
	CUDA_CHECK(cudaStreamSynchronize(stream));
	handle.getDeviceAllocator()->deallocate(P_sum_, sizeof(float), stream);

	return (P_sum > FLT_EPSILON) ? P_sum : FLT_EPSILON;
}


__global__ void
__attractive_fast(const float *__restrict__ VAL,
                    const int *__restrict__ COL,
                    const int *__restrict__ ROW,
                    const float *__restrict__ Y,
                    const float *__restrict__ norm,
                    const float *__restrict__ norm_add1,
                    float *__restrict__ attract, const int NNZ,
                    const int n, const int dim) {
    // Notice attract, Y and repel are all F-contiguous
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < NNZ) {
        const int i = ROW[index], j = COL[index];

        float euclidean_d = 0.0f;
        for (int k = 0; k < dim; k++)
            euclidean_d += (Y[k*n + i] * Y[k*n + j]);

        const float PQ = __fdividef(VAL[index], (norm[i] + norm_add1[j] - 2.0f*euclidean_d ) ); // 1/x

        for (int k = 0; k < dim; k++)
            atomicAdd(&attract[k*n + i],     PQ * (Y[k*n + i] - Y[k*n + j]) );
    }
}
__global__ void
__attractive_fast_2dim(const float *__restrict__ VAL,
	                    const int *__restrict__ COL,
	                    const int *__restrict__ ROW,
	                    const float *__restrict__ Y1,
	                    const float *__restrict__ Y2,
	                    const float *__restrict__ norm,
	                    const float *__restrict__ norm_add1,
	                    float *__restrict__ attract1,
	                    float *__restrict__ attract2,
	                    const int NNZ, const int n, const int dim) {
    // Notice attract, Y and repel are all F-contiguous
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < NNZ) {
        const int i = ROW[index], j = COL[index];
        const float PQ = __fdividef(VAL[index] , (norm[i] + norm_add1[j] -2.0f*(Y1[i]*Y1[j] + Y2[i]*Y2[j]) ) ); // 1/x

        atomicAdd(&attract1[i],     PQ * (Y1[i] - Y1[j]) );
        atomicAdd(&attract2[i],     PQ * (Y2[i] - Y2[j]) );
    }
}
void attractive_fast(const float *__restrict__ VAL,
                    const int *__restrict__ COL,
                    const int *__restrict__ ROW,
                    const float *__restrict__ Y,
                    const float *__restrict__ norm,
                    const float *__restrict__ norm_add1,
                    float *__restrict__ attract, const int NNZ,
                    const int n, const int dim,
                    cudaStream_t stream,
                    const int gridSize, const int blockSize) {
    CUDA_CHECK(cudaMemsetAsync(attract, 0, sizeof(float) * n * dim, stream));

    if (dim == 2)
    	__attractive_fast_2dim<<<gridSize, blockSize, 0, stream>>>(VAL, COL, ROW, Y,
        	Y + n, norm, norm_add1, attract, attract + n, NNZ, n, dim);
    else
    	__attractive_fast<<<gridSize, blockSize, 0, stream>>>(VAL, COL, ROW, Y,
        	norm, norm_add1, attract, NNZ, n, dim);
    CUDA_CHECK(cudaPeekAtLastError());
}



__global__ void
__repulsive_fast(const float *__restrict__ Y,
                float *__restrict__ repel,
                const float *__restrict__ norm,
                const float *__restrict__ norm_add1,
                float *__restrict__ sum_Z,
                const int n, const int dim)
{
    const int j = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every item in row
    const int i = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every row

    if (j > i && i < n && j < n) {
        float euclidean_d = 0.0f;
        for (int k = 0; k < dim; k++)
            euclidean_d += (Y[k*n + i] * Y[k*n + j]);

        const float Q = __fdividef(1.0f, (norm[i] + norm_add1[j] - 2.0f*euclidean_d) ); // 1/x
        const float Q2 = Q*Q;

        for (int k = 0; k < dim; k++) {
            const float force = Q2 * (Y[k*n + i] - Y[k*n + j]);

            atomicAdd(&repel[k*n + i],  - force);  // repel[k*n + i] -= force
            atomicAdd(&repel[k*n + j],  force);    // repel[k*n + j] += force
        }
        atomicAdd(&sum_Z[i], Q); // Z += Q
    }
}
__global__ void
__repulsive_fast_2dim(const float *__restrict__ Y1,
					const float *__restrict__ Y2,
	                float *__restrict__ repel1a,
	                float *__restrict__ repel1b,
	                float *__restrict__ repel2a,
	                float *__restrict__ repel2b,
	                const float *__restrict__ norm,
	                const float *__restrict__ norm_add1,
	                float *__restrict__ sum_Z,
	                const int n, const int dim)
{
    const int j = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every item in row
    const int i = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every row
    if (j > i && i < n && j < n) {
        const float Q = __fdividef(1.0f, (norm[i] + norm_add1[j] -2.0f*(Y1[i]*Y1[j] + Y2[i]*Y2[j]) )  ); // 1/x
        const float Q2 = Q*Q;

        const float force1 = Q2 * (Y1[i] - Y1[j]);
        repel1a[i] -= force1;
        repel1b[j] += force1;

        const float force2 = Q2 * (Y2[i] - Y2[j]);
        repel2a[i] -= force2;
        repel2b[j] += force2;

        atomicAdd(&sum_Z[i], Q); // Z += Q
    }
}


template <int TPB_X = 32, int TPB_Y = 32>
float repulsive_fast(const float *__restrict__ Y,
                    float *__restrict__ repel,
                    const float *__restrict__ norm,
                    const float *__restrict__ norm_add1,
                    float *__restrict__ sum_Z,
                    const int n, const int dim,
                    cudaStream_t stream,
                    const dim3 threadsPerBlock, const dim3 numBlocks)
{
    CUDA_CHECK(cudaMemsetAsync(sum_Z, 0, sizeof(float) * n, stream));
    CUDA_CHECK(cudaMemsetAsync(repel, 0, sizeof(float) * n * dim * 2, stream));

    if (dim == 2) {
    	__repulsive_fast_2dim<<<numBlocks, threadsPerBlock, 0, stream>>>(
    		Y, Y + n, repel, repel + n, repel + 2*n, repel + 3*n, norm, norm_add1, sum_Z, n, dim);
    	CUDA_CHECK(cudaPeekAtLastError());

    	MLCommon::LinAlg::eltwiseAdd(repel, repel, repel + n, n, stream);
    	MLCommon::LinAlg::eltwiseAdd(repel + n, repel + 2*n, repel + 3*n, n, stream);
    }
    else {
    	__repulsive_fast<<<numBlocks, threadsPerBlock, 0, stream>>>(
    		Y, repel, norm, norm_add1, sum_Z, n, dim);
    	CUDA_CHECK(cudaPeekAtLastError());
    }
    
    thrust::device_ptr<float> begin = thrust::device_pointer_cast(sum_Z);
    double Z = (double) thrust::reduce(thrust::cuda::par.on(stream), begin, begin + n);
    return 1.0f / (Z * 2.0f);
}



__global__ void 
__apply_forces(const float *__restrict__ attract,
				 const float *__restrict__ repel,
				 float *__restrict__ Y, float *__restrict__ iY,
				 float *__restrict__ gains, const int n,
				 const int dim, const float Z, const float min_gain,
				 const float momentum, const float eta,
				 float *__restrict__ means, const float SIZE,
				 double *__restrict__ dY, const bool add_grad) {
	// Everything is F-Contiguous
	const int i = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every item in column
	if (i < SIZE) {
		const float dy = attract[i] + Z * repel[i];

		if (signbit(dy) != signbit(iY[i]))
			gains[i] += 0.2f;
		else
			gains[i] *= 0.8f;

		if (gains[i] < min_gain)
			gains[i] = min_gain;

		iY[i] = momentum * iY[i] - eta * (gains[i] * dy);
		Y[i] += iY[i];
		// Also find mean
		atomicAdd(&means[i / n], Y[i]);
		// Add grad norm for early stopping
		if (add_grad == true)
			atomicAdd(&dY[i / n], dy*dy);
	}
}

template <int TPB_X = 32, int TPB_Y = 32>
double apply_forces(const float *__restrict__ attract,
				float *__restrict__ means,
				const float *__restrict__ repel, float *__restrict__ Y,
				float *__restrict__ iY, float *__restrict__ gains, 
				const int n, const int dim, const float Z, 
				const float min_gain, const float momentum,
				const float eta, cudaStream_t stream,
				const int gridSize, const int blockSize,
				double *__restrict__ dY, const bool add_grad) {

	CUDA_CHECK(cudaMemsetAsync(means, 0, sizeof(float) * dim, stream));
	if (add_grad)
		CUDA_CHECK(cudaMemsetAsync(dY, 0, sizeof(double) * n, stream));

	__apply_forces<<<gridSize, blockSize, 0, stream>>>(
		attract, repel, Y, iY, gains, n, dim, Z, min_gain, momentum, eta, means, n*dim,
		dY, add_grad);
	CUDA_CHECK(cudaPeekAtLastError());

	// Divide by 1/n
	MLCommon::LinAlg::scalarMultiply(means, (const float*) means, 1.0f/n, dim, stream);

	// Subtract by mean
	MLCommon::LinAlg::matrixVectorOp(
	    Y, Y, means, dim, n, false, true,
	    [] __device__(float a, float b) {
	    	return a - b;
	    }, stream);

	// Early stopping gradient norm
	double grad_norm = INFINITY;
	if (add_grad) {
		thrust::device_ptr<double> begin = thrust::device_pointer_cast(dY);
    	grad_norm = sqrt(thrust::reduce(thrust::cuda::par.on(stream), begin, begin + n));
	}
	return grad_norm;
}


}
}