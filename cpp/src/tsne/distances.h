
#pragma once

#include <linalg/eltwise.h>
#include <selection/knn.h>
#include "cuML.hpp"
#include "utils.h"

namespace ML {
using namespace MLCommon;


void
get_distances(const float *X, const int n, const int p, long *indices,
			float *distances, const int n_neighbors,
			cudaStream_t stream)
{
	assert(X != NULL);

	float **knn_input = new float *[1];
	int *sizes = new int[1];
	knn_input[0] = (float *)X;
	sizes[0] = n;

	MLCommon::Selection::brute_force_knn(knn_input, sizes, 1, p,
										const_cast<float *>(X), n, indices,
										distances, n_neighbors, stream);
	delete knn_input, sizes;
}



float
normalize_distances(const int n, float *distances, const int n_neighbors,
					cudaStream_t stream)
{
	// Now D / max(abs(D)) to allow exp(D) to not explode
	assert(distances != NULL);
	thrust_t<float> begin = to_thrust(distances);

	float maxNorm = MAX(*(thrust::max_element(__STREAM__, begin, begin + n*n_neighbors)),
						*(thrust::min_element(__STREAM__, begin, begin + n*n_neighbors)));
	if (maxNorm == 0.0f) maxNorm = 1.0f;

	// Divide distances inplace by max
	const div = 1.0f/maxNorm;
	array_multiply(distances, n*n_neighbors, div, stream);
	return div;
}



template <int TPB_X = 32>
void symmetrize_perplexity(float *P, long *indices, COO_t<float> *P_PT,
						 const int n, const int k, const float P_sum,
						 const float exaggeration, cudaStream_t stream,
						 const cumlHandle &handle) {
	assert(P != NULL && indices != NULL);

	// Convert to COO
	COO_t<float> P_COO;
	COO_t<float> P_PT_with_zeros;
	Sparse::from_knn(indices, P, n, k, &P_COO);
	handle.getDeviceAllocator()->deallocate(P, sizeof(float) * n * k, stream);
	handle.getDeviceAllocator()->deallocate(indices, sizeof(long) * n * k, stream);

	// Perform (P + P.T) / P_sum * early_exaggeration
	const float div = exaggeration / (2.0f * P_sum);
	array_multiply(P_COO.vals, P_COO.nnz, div, stream);

	// Symmetrize to form P + P.T
	Sparse::coo_symmetrize<TPB_X, float>(
		&P_COO, &P_PT_with_zeros,
		[] __device__(int row, int col, float val, float trans) {
			return val + trans;
		},
		stream);
	P_COO.destroy();

	// Remove all zeros in P + PT
	Sparse::coo_sort<float>(&P_PT_with_zeros, stream);

	Sparse::coo_remove_zeros<TPB_X, float>(&P_PT_with_zeros, P_PT, stream);
	P_PT_with_zeros.destroy();
}


}  // namespace ML
