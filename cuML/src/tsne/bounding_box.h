
using namespace ML;
#include "utils.h"
#include "cuda_utils.h"

#pragma once

//
namespace BoundingBox_ {


__global__
__launch_bounds__(THREADS1, FACTOR1)
void boundingBoxKernel(const int N_NODES, 
                        const int n, 
                        volatile int * __restrict__ startd, 
                        volatile int * __restrict__ childd, 
                        volatile float * __restrict__ massd, 
                        volatile float * __restrict__ posxd, 
                        volatile float * __restrict__ posyd, 
                        volatile float * __restrict__ maxxd, 
                        volatile float * __restrict__ maxyd, 
                        volatile float * __restrict__ minxd, 
                        volatile float * __restrict__ minyd) 
{
	int i, j, k, inc;
	float val, minx, maxx, miny, maxy;
	__shared__ volatile float sminx[THREADS1], smaxx[THREADS1], sminy[THREADS1], smaxy[THREADS1];

	// initialize with valid data (in case #bodies < #threads)
	minx = maxx = posxd[0];
	miny = maxy = posyd[0];

	// scan all bodies
	i = threadIdx.x;
	inc = THREADS1 * gridDim.x;
	for (j = i + blockIdx.x * THREADS1; j < n; j += inc) {
		val = posxd[j];
		minx = MIN(minx, val);
		maxx = MAX(maxx, val);
		val = posyd[j];
		miny = MIN(miny, val);
		maxy = MAX(maxy, val);
	}

	// reduction in shared memory
	sminx[i] = minx;
	smaxx[i] = maxx;
	sminy[i] = miny;
	smaxy[i] = maxy;

	for (j = THREADS1 / 2; j > 0; j /= 2) {
		kernel_sync();
		if (i < j) {
			k = i + j;
			sminx[i] = minx = MIN(minx, sminx[k]);
			smaxx[i] = maxx = MAX(maxx, smaxx[k]);
			sminy[i] = miny = MIN(miny, sminy[k]);
			smaxy[i] = maxy = MAX(maxy, smaxy[k]);
		}
	}

	// write block result to global memory
	if (i == 0) {
		k = blockIdx.x;
		minxd[k] = minx;
		maxxd[k] = maxx;
		minyd[k] = miny;
		maxyd[k] = maxy;
		kernel_fence();

		inc = gridDim.x - 1;
		if (inc == atomicInc(&blkcntd, inc)) {
			// I'm the last block, so combine all block results
			for (j = 0; j <= inc; j++) {
				minx = MIN(minx, minxd[j]);
				maxx = MAX(maxx, maxxd[j]);
				miny = MIN(miny, minyd[j]);
				maxy = MAX(maxy, maxyd[j]);
			}

			// compute 'radius'
			radiusd = MAX(maxx - minx, maxy - miny) * 0.5f + 1e-5f;

			// create root node
			k = N_NODES;
			bottomd = k;

			massd[k] = -1.0f;
			startd[k] = 0;
			posxd[k] = (minx + maxx) * 0.5f;
			posyd[k] = (miny + maxy) * 0.5f;
			k *= 4;
			for (i = 0; i < 4; i++)
				childd[k + i] = -1;

			stepd++;
		}
	}
}

// end namespace
}