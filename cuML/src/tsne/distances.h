
// From UMAP/knn_graph/algo.h

#include <iostream>
#include <cuda_utils.h>
#include "utils.h"

#include "knn/knn.h"
#include "linalg/eltwise.h"

#pragma once
using namespace ML;
#include <math.h>


// TODO convert to CUDA kernel to find max(abs(D))
template <typename Type>
Type maxAbs(const Type * __restrict__ x, const int p) {
	register Type max = 0, temp;
	for (register int i = 0; i < p; i++)
		if ( (temp = fabs(x[i])) > max)
			max = temp;
	return max;
}


template <typename Type>
void getDistances(	const Type * __restrict__ X,
					const int n,
					const int p,
					long *indices,
					Type *distances,
					const int n_neighbors,
					cudaStream_t stream)
{
	kNNParams *params = new kNNParams[1];
	params[0].ptr = X;
	params[1].N = n;
	knn = new KNN(p);

	knn->fit(params, 1);
	knn->search(X, n, indices, distances, n_neighbors);
	// No need to postprocess distances since it's squared!


	// Now D / max(abs(D)) to allow exp(D) to not explode
	//// TODO convert to GPU code
	Type *max = (Type*) malloc(sizeof(Type) * n);

	#pragma omp parallel for if (n > 100) default(none)
	for (int i = 0; i < n * n_neighbors; i++)
		max[i] = maxAbs(distances + i*n, p);

	Type maxNorm = 0;
	for (int i = 0; i < n; i++)
		if (max[i] > maxNorm) maxNorm = max[i];
	free(max);
	////

	// Divide distances inplace by max
	Type div_maxNorm = 1.0f/maxNorm; // Mult faster than div
	scalarMultiply(distances, distances, div_maxNorm, n*n_neighbors, stream);


	// Remove temp variables
	delete knn;
	delete params;
}