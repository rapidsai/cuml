
#pragma once

#include <float.h>
#include <math.h>
#include "utils.h"
#define ceil(a, b) ((a + b - 1) / b)

namespace ML {

using namespace ML;
using namespace MLCommon;



__global__ void
__form_t_distribution(float *__restrict__ Q,
					const float *__restrict__ norm,
					const int n, float *__restrict__ sum_Q) {
	const int j = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every item in row
	const int i = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every row

	if (i < n && j < n) {
		if (i == j)
			Q[i*n + j] = 0.0f;
		else if (j > i) {
			Q[j*n + i] = Q[i*n + j] = 1.0f / (Q[i*n + j] + norm[i] + norm[j] + 1.0f);
			atomicAdd(&sum_Q[i], Q[i*n + j]);
		}
	}
}

template <int TPB_X = 32, int TPB_Y = 32>
double form_t_distribution(float *__restrict__ Q, const float *__restrict__ norm,
							const int n, float *__restrict__ sum_Q,
							double *__restrict__ sum, cudaStream_t stream) {
	cudaMemset(sum_Q, 0, sizeof(float) * n);

	static const dim3 threadsPerBlock(TPB_X, TPB_Y);
	const dim3 numBlocks(ceil(n, threadsPerBlock.x), ceil(n, threadsPerBlock.y));

	__form_t_distribution<<<numBlocks, threadsPerBlock, 0, stream>>>(Q, norm, n, sum_Q);
	CUDA_CHECK(cudaPeekAtLastError());

	double Z = (double) thrust::reduce(__STREAM__, sum_Q, sum_Q + n);
	return 1.0f / (2.0f * Z);
}

__global__ void
__attractive_forces(const float *__restrict__ VAL,
					const int *__restrict__ COL,
					const int *__restrict__ ROW,
					const float *__restrict__ Q,
					const float *__restrict__ Y,
					float *__restrict__ attract, const int NNZ,
					const int n, const int K) {
	// Notice attract, Y and repel are all F-contiguous
	const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < NNZ) {
		const int i = ROW[index];
		const int j = COL[index];
		const float PQ = VAL[index] * ((j > i) ? Q[i*n + j] : Q[j*n + i]);
		for (int l = 0; l < K; l++)
			// attract[i*K + j] += PQ * (Y[i, j] - Y[j, j]);
			atomicAdd(&attract[l*n + i], PQ * (Y[l*n + i] - Y[l*n + j]));
	}
}
void attractive_forces(const float *__restrict__ VAL,
					 const int *__restrict__ COL, const int *__restrict__ ROW,
					 const float *__restrict__ Q, const float *__restrict__ Y,
					 float *__restrict__ attract, const int NNZ, const int n,
					 const int K, cudaStream_t stream) {
	cudaMemset(attract, 0, sizeof(float) * n * K);
	__attractive_forces<<<ceil(NNZ, 1024), 1024, 0, stream>>>(VAL, COL, ROW, Q, Y,
		attract, NNZ, n, K);
    CUDA_CHECK(cudaPeekAtLastError());
}



__global__ void 
__postprocess_Q(float *__restrict__ Q, float *__restrict__ sum_Q, const int n) {
	const int j = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every item in row
	const int i = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every row
	if (i < n && j < n) {
		float q = Q[i*n + j]; q *= q;
		atomicAdd(&sum_Q[i], q);
	}
}

template <int TPB_X = 32, int TPB_Y = 32>
void postprocess_Q(float *__restrict__ Q, float *__restrict__ sum_Q,
					const int n, cudaStream_t stream) {
	cudaMemset(sum_Q, 0, sizeof(float) * n);
	const dim3 threadsPerBlock(TPB_X, TPB_Y);
	const dim3 numBlocks(ceil(n, threadsPerBlock.x), ceil(n, threadsPerBlock.y));
	__postprocess_Q<<<numBlocks, threadsPerBlock, 0, stream>>>(Q, sum_Q, n);
	CUDA_CHECK(cudaPeekAtLastError());
}



__global__ void 
__repel_minus_QY(float *__restrict__ repel,
				 const float *__restrict__ neg_sum_Q,
				 const float *__restrict__ Y, const int n,
				 const int K) {
	const int j = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every column
	const int i = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every item in column
	if (j < K && i < n)
		// repel[i*n + j] -= Q_sum[i] * Y[i*n + j];
		atomicAdd(&repel[j*n + i], neg_sum_Q[i] * Y[j*n + i]);  // Y, repel is F-Contiguous
}

template <int TPB_X = 32, int TPB_Y = 32>
void repel_minus_QY(float *__restrict__ repel, float *__restrict__ sum_Q,
					const float *__restrict__ Y, const int n, const int K,
					cudaStream_t stream) {
	thrust::transform(__STREAM__, sum_Q, sum_Q + n, sum_Q, -1 * _1);

	static const dim3 threadsPerBlock(TPB_X, TPB_Y);
	const dim3 numBlocks(ceil(K, threadsPerBlock.x), ceil(n, threadsPerBlock.y));

	__repel_minus_QY<<<numBlocks, threadsPerBlock, 0, stream>>>(repel, sum_Q, Y, n, K);
	CUDA_CHECK(cudaPeekAtLastError());
}



__global__ void
__get_norm_slow(const float *__restrict__ Y, float *__restrict__ norm, 
			const int n, const int n_components)
{
	const int i = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every item in col
	const int j = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every col
	if (i < n && j < n_components)
		atomicAdd(&norm[i], Y[j*n + i] * Y[j*n + i]);
}

template <int TPB_X = 32, int TPB_Y = 32>
void get_norm_slow(const float *__restrict__ Y, float *__restrict__ norm,
			  const int n, const int n_components, cudaStream_t stream) {
	// Notice Y is F-Contiguous
	cudaMemset(norm, 0, sizeof(float) * n);

	static const dim3 threadsPerBlock(TPB_X, TPB_Y);
	const dim3 numBlocks(ceil(n, threadsPerBlock.x), ceil(n_components, threadsPerBlock.y));
	__get_norm<<<numBlocks, threadsPerBlock, 0, stream>>>(Y, norm, n, n_components);
	CUDA_CHECK(cudaPeekAtLastError());
}


__global__ void 
__apply_forces_slow(const float *__restrict__ attract,
				 const float *__restrict__ repel,
				 float *__restrict__ Y, float *__restrict__ iY,
				 float *__restrict__ gains, const int n,
				 const int K, const double Z, const float min_gain,
				 const float momentum, const float eta) {
	// Everything is F-Contiguous
	const int j = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every column
	const int i = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every item in column
	if (j < K && i < n) {
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
void apply_forces_slow(const float *__restrict__ attract,
				const float *__restrict__ repel, float *__restrict__ Y,
				float *__restrict__ iY, float *__restrict__ gains, 
				const int n, const int K, const float Z, 
				const float min_gain, const float momentum,
				const float eta, cudaStream_t stream) {
	static const dim3 threadsPerBlock(TPB_X, TPB_Y);
	const dim3 numBlocks(ceil(K, threadsPerBlock.x), ceil(n, threadsPerBlock.y));

	__apply_forces<<<numBlocks, threadsPerBlock, 0, stream>>>(
		attract, repel, Y, iY, gains, n, K, Z, min_gain, momentum, eta);
	CUDA_CHECK(cudaPeekAtLastError());
}


}  // namespace ML
