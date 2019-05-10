
// From UMAP/knn_graph/algo.h

using namespace ML;
#include "utils.h"

#include "knn/knn.h"
#include "linalg/eltwise.h"

#pragma once


namespace Distances_ {

// FAISS returns d^2 not d.
template <typename Type>
void getDistances(	const Type * __restrict__ X,
					const int n,
					const int p,
					long *indices,
					float *distances,
					const int n_neighbors,
					const int SIZE,
					cudaStream_t stream)
{
	kNNParams *params = new kNNParams[1];
	params[0].ptr = X;
	params[1].N = n;
	knn = new KNN(p);

	knn->fit(params, 1);
	knn->search(X, n, indices, distances, n_neighbors);
	// No need to postprocess distances since it's already L2 squared!


	// Now D / max(abs(D)) to allow exp(D) to not explode
	// Max(abs(D)) == max(min(D), max(D))
	thrust::device_ptr<const float> begin = thrust::device_pointer_cast(distances);
    thrust::device_ptr<const float> end = begin + SIZE;

    float maxNorm = MAX(Utils_::max_array(begin, end, stream), 
    					Utils_::min_array(begin, end, stream));


	// Divide distances inplace by max
	float div_maxNorm = 1.0f/maxNorm; // Mult faster than div
	LinAlg::scalarMultiply(distances, distances, div_maxNorm, SIZE, stream);


	// Remove temp variables
	delete knn;
	delete params;
}


// end namespace
}