
#pragma once

#include <float.h>
#include <math.h>
#include "utils.h"
#define ceil(a, b) ((a + b - 1) / b)

namespace ML {

using namespace ML;
using namespace MLCommon;


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
		float sum_P_row, sum_disti_Pi, div_sum_Pi;
		float entropy, entropy_diff;
		register int ik = i * k;

		for (int step = 0; step < epochs; step++) {
			sum_Pi = FLT_EPSILON;
			sum_disti_Pi = 0;
			sum_P_row = 0;

			// Exponentiate to get guassian
			for (int j = 0; j < k; j++) {
				P[ik + j] = __expf(-distances[ik + j] * beta);
				sum_Pi += P[ik + j];
			}

			// Normalize
			div_sum_Pi = 1.0f / sum_Pi;
			for (int j = 0; j < k; j++) {
				P[ik + j] *= div_sum_Pi;  // P[i*k + j] / sum_Pi
				sum_disti_Pi += distances[ik + j] * P[ik + j];
				sum_P_row += P[ik + j];
			}

			entropy = __logf(sum_Pi) + beta * sum_disti_Pi;
			entropy_diff = entropy - desired_entropy;
			if (fabs(entropy_diff) <= tol) 
				break;

			// Bisection search
			if (entropy_diff > 0) {
				beta_min = beta;
				if (isinf(beta_max))
					beta *= 2.0f;
				else
					beta = (beta + beta_max) * 0.5f;
			} else {
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
					 const int k, cudaStream_t stream) {
	const float desired_entropy = logf(perplexity);
	float *P_sum_, P_sum;
	cudaMalloc(&P_sum_, sizeof(float));
	cudaMemset(P_sum_, 0, sizeof(float));

	__determine_sigmas<<<ceil(n, 1024), 1024, 0, stream>>>(
		distances, P, perplexity, desired_entropy, P_sum_, epochs, tol, n, k);

	cudaMemcpy(&P_sum, P_sum_, sizeof(float), cudaMemcpyDeviceToHost);
	CUDA_CHECK(cudaPeekAtLastError());
	return P_sum;
}




__global__ void
__get_norm_fast(const float *__restrict__ Y, float *__restrict__ norm, 
			const int n, const int dim)
{
	const int i = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every item in col
	const int j = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every col
	if (i < n && j < dim)
		atomicAdd(&norm[i], Y[j*n + i] * Y[j*n + i]);
}

template <int TPB_X = 32, int TPB_Y = 32>
void get_norm_fast(const float *__restrict__ Y, float *__restrict__ norm,
			  const int n, const int dim, cudaStream_t stream) {
	// Notice Y is F-Contiguous
	cudaMemset(norm, 0, sizeof(float) * n);

	static const dim3 threadsPerBlock(TPB_X, TPB_Y);
	const dim3 numBlocks(ceil(n, threadsPerBlock.x), ceil(dim, threadsPerBlock.y));
	__get_norm_fast<<<numBlocks, threadsPerBlock, 0, stream>>>(Y, norm, n, dim);
	CUDA_CHECK(cudaPeekAtLastError());
}



__global__ void
__attractive_fast(const float *__restrict__ VAL,
                    const int *__restrict__ COL,
                    const int *__restrict__ ROW,
                    const float *__restrict__ Y,
                    const float *__restrict__ norm,
                    float *__restrict__ attract, const int NNZ,
                    const int n, const int dim) {
    // Notice attract, Y and repel are all F-contiguous
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < NNZ) {
        const int i = ROW[index];
        const int j = COL[index];

        float d = 0.0f;
        for (int k = 0; k < dim; k++)
            //d += Y[i, k] * Y[j, k]
            d += (Y[k*n + i] * Y[k*n + j]);

        const float PQ = VAL[index] / (1.0f - 2.0f*d + norm[i] + norm[j]);

        for (int k = 0; k < dim; k++)
            atomicAdd(&attract[k*n + i],     PQ * (Y[k*n + i] - Y[k*n + j]) );
            // attract[i*K + j] += PQ * (Y[i, j] - Y[j, j]);
    }
}
__global__ void
__attractive_fast_2dim(const float *__restrict__ VAL,
	                    const int *__restrict__ COL,
	                    const int *__restrict__ ROW,
	                    const float *__restrict__ Y1,
	                    const float *__restrict__ Y2,
	                    const float *__restrict__ norm,
	                    float *__restrict__ attract1,
	                    float *__restrict__ attract2, const int NNZ,
	                    const int n, const int dim) {
    // Notice attract, Y and repel are all F-contiguous
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < NNZ) {
        const int i = ROW[index];
        const int j = COL[index];

        const float PQ = VAL[index] / \
        	(1.0f - 2.0f*(Y1[i]*Y1[j] + Y2[i]*Y2[j]) + norm[i] + norm[j]);

        atomicAdd(&attract1[i],     PQ * (Y1[i] - Y1[j]) );
        atomicAdd(&attract2[i],     PQ * (Y2[i] - Y2[j]) );
    }
}
template <int gridSize, int blockSize>
void attractive_fast(const float *__restrict__ VAL,
                    const int *__restrict__ COL,
                    const int *__restrict__ ROW,
                    const float *__restrict__ Y,
                    const float *__restrict__ norm,
                    float *__restrict__ attract, const int NNZ,
                    const int n, const int dim,
                    cudaStream_t stream) {
    cudaMemset(attract, 0, sizeof(float) * n * dim);
    
    if (dim == 2)
    	__attractive_fast_2dim<<<gridSize, blockSize, 0, stream>>>(VAL, COL, ROW, Y,
        	Y + n, norm, attract, attract + n, NNZ, n, dim);
    else
    	__attractive_fast<<<ceil(NNZ, 1024), 1024, 0, stream>>>(VAL, COL, ROW, Y,
        	norm, attract, NNZ, n, dim);
    CUDA_CHECK(cudaPeekAtLastError());
}



__global__ void
__repulsive_fast(const float *__restrict__ Y,
                float *__restrict__ repel,
                const float *__restrict__ norm,
                float *__restrict__ sum_Z,
                const int n, const int dim)
{
    const int j = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every item in row
    const int i = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every row

    if (j > i && i < n && j < n) {
        float d = 0.0f;
        for (int k = 0; k < dim; k++)
            //d += Y[i, k] * Y[j, k]
            d += (Y[k*n + i] * Y[k*n + j]);

        const float Q = 1.0f  /  (1.0f - 2.0f*d  + norm[i] + norm[j]);
        atomicAdd(&sum_Z[i], Q); // Z += Q
        const float Q2 = Q*Q;

        for (int k = 0; k < dim; k++) {
            const float force = Q2 * (Y[k*n + i] - Y[k*n + j]);
            // repel = Q2 * (Y[i, k] - Y[j, k]);

            atomicAdd(&repel[k*n + i],  - force);  // repel[k*n + i] -= force
            atomicAdd(&repel[k*n + j],  force);    // repel[k*n + j] += force
        }
    }
}
__global__ void
__repulsive_fast_2dim(const float *__restrict__ Y1,
				const float *__restrict__ Y2,
                float *__restrict__ repel1,
                float *__restrict__ repel2,
                const float *__restrict__ norm,
                float *__restrict__ sum_Z,
                const int n, const int dim)
{
    const int j = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every item in row
    const int i = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every row

    if (j > i && i < n && j < n) {

        const float Q = 1.0f / \
        	(1.0f - 2.0f*(Y1[i]*Y1[j] + Y2[i]*Y2[j]) + norm[i] + norm[j]);

        atomicAdd(&sum_Z[i], Q); // Z += Q
        const float Q2 = Q*Q;

        const float force1 = Q2 * (Y1[i] - Y1[j]);
        atomicAdd(&repel1[i],  - force1);
        atomicAdd(&repel1[j],  force1);

        const float force2 = Q2 * (Y2[i] - Y2[j]);
        atomicAdd(&repel2[i],  - force2);
        atomicAdd(&repel2[j],  force2);
    }
}


template <int TPB_X = 32, int TPB_Y = 32>
float repulsive_fast(const float *__restrict__ Y,
                    float *__restrict__ repel,
                    const float *__restrict__ norm,
                    float *__restrict__ sum_Z,
                    const int n, const int dim,
                    cudaStream_t stream)
{
    cudaMemset(sum_Z, 0, sizeof(float) * n);
    cudaMemset(repel, 0, sizeof(float) * n * dim);

    const dim3 threadsPerBlock(TPB_X, TPB_Y);
    const dim3 numBlocks(ceil(n, threadsPerBlock.x), ceil(n, threadsPerBlock.y));

    if (dim == 2)
    	__repulsive_fast_2dim<<<numBlocks, threadsPerBlock, 0, stream>>>(
    		Y, Y + n, repel, repel + n, norm, sum_Z, n, dim);
    else
    	__repulsive_fast<<<numBlocks, threadsPerBlock, 0, stream>>>(
    		Y, repel, norm, sum_Z, n, dim);
    CUDA_CHECK(cudaPeekAtLastError());

    thrust_t<float> begin = to_thrust(sum_Z);
    double Z = (double) thrust::reduce(__STREAM__, begin, begin + n);
    return 1.0f / (2.0f * Z) + FLT_EPSILON;
}



__global__ void
__find_mean_fast(const float * __restrict__ Y, float * __restrict__ means, 
			const int n, const int dim) {
	// Y is F-Contiguous
	const int i = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every item in col
	const int j = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every col
	if (i < n && j < dim) atomicAdd(&means[j], Y[j*n + i]);
}
__global__ void
__subtract_mean_fast(float * __restrict__ Y, const float * __restrict__ means, 
			const int n, const int dim) {
	// Y is F-Contiguous
	const int i = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every item in col
	const int j = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every col
	if (i < n && j < dim) Y[j*n + i] -= means[j];
}

template <int TPB_X = 32, int TPB_Y = 32>
void remove_mean_fast(float *__restrict__ Y, float *__restrict__ means,
			  		const int n, const int dim, cudaStream_t stream) {
	// Notice Y is F-Contiguous
	cudaMemset(means, 0, sizeof(float) * dim);

	static const dim3 threadsPerBlock(TPB_X, TPB_Y);
	const dim3 numBlocks(ceil(n, threadsPerBlock.x), ceil(dim, threadsPerBlock.y));
	__find_mean_fast<<<numBlocks, threadsPerBlock, 0, stream>>>(Y, means, n, dim);
	CUDA_CHECK(cudaPeekAtLastError());

	// Divide by 1/n
	const float div = 1.0f / n;
	thrust_t<float> begin = to_thrust(means);
	thrust::transform(__STREAM__, begin, begin + dim, begin, div * _1);

	// Subtract the mean
	__subtract_mean_fast<<<numBlocks, threadsPerBlock, 0, stream>>>(Y, means, n, dim);
	CUDA_CHECK(cudaPeekAtLastError());
}



__global__ void 
__apply_forces(const float *__restrict__ attract,
				 const float *__restrict__ repel,
				 float *__restrict__ Y, float *__restrict__ iY,
				 float *__restrict__ gains, const int n,
				 const int dim, const float Z, const float min_gain,
				 const float momentum, const float eta) {
	// Everything is F-Contiguous
	const int j = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every column
	const int i = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every item in column
	if (j < dim && i < n) {
		const int index = j*n + i;
		const float dy = attract[index] + Z * repel[index];

		if (signbit(dy) != signbit(iY[index]))
			gains[index] += 0.2f;
		else
			gains[index] *= 0.8f;

		if (gains[index] < min_gain)
			gains[index] = min_gain;

		iY[index] = momentum * iY[index] - eta * (gains[index] * dy);
		Y[index] += iY[index];
	}
}

template <int TPB_X = 32, int TPB_Y = 32>
void apply_forces(const float *__restrict__ attract,
				float *__restrict__ means,
				const float *__restrict__ repel, float *__restrict__ Y,
				float *__restrict__ iY, float *__restrict__ gains, 
				const int n, const int dim, const float Z, 
				const float min_gain, const float momentum,
				const float eta, cudaStream_t stream) {
	static const dim3 threadsPerBlock(TPB_X, TPB_Y);
	const dim3 numBlocks(ceil(dim, threadsPerBlock.x), ceil(n, threadsPerBlock.y));

	__apply_forces<<<numBlocks, threadsPerBlock, 0, stream>>>(
		attract, repel, Y, iY, gains, n, dim, Z, min_gain, momentum, eta);
	CUDA_CHECK(cudaPeekAtLastError());

	// Find mean and remove it
	remove_mean_fast(Y, means, n, dim, stream);
}


}