
using namespace ML;
#include "utils.h"
#include "cuda_utils.h"

#pragma once


__global__
void boundingBoxKernel(volatile int * __restrict__ cell_starts, 
                       volatile int * __restrict__ children, 
                       volatile float * __restrict__ cell_mass, 
                       volatile float * __restrict__ embedding_x, 
                       volatile float * __restrict__ embedding_y, 
                       volatile float * __restrict__ x_max, 
                       volatile float * __restrict__ y_max, 
                       volatile float * __restrict__ x_min, 
                       volatile float * __restrict__ y_min,
                       const int N_NODES,
                       const int n,
                       const int bounding_box_threads)
{
    register int i, j, k, inc;
    register float val, minx, maxx, miny, maxy;

    extern __shared__ float bounding_shared_memory[];
    float* x_min_shared = bounding_shared_memory;
    float* x_max_shared = x_min_shared + bounding_box_threads;
    float* y_min_shared = x_max_shared + bounding_box_threads;
    float* y_max_shared = y_min_shared + bounding_box_threads;
   
    // initialize with valid data (in case #bodies < #threads)
    minx = maxx = embedding_x[0];
    miny = maxy = embedding_y[0];

    // scan all bodies
    i = threadIdx.x;
    inc = bounding_box_threads * gridDim.x;
    for (j = i + blockIdx.x * bounding_box_threads; j < n; j += inc) {
        val = embedding_x[j];
        minx = fminf(minx, val);
        maxx = fmaxf(maxx, val);
        val = embedding_y[j];
        miny = fminf(miny, val);
        maxy = fmaxf(maxy, val);
    }

    // reduction in shared memory
    x_min_shared[i] = minx;
    x_max_shared[i] = maxx;
    y_min_shared[i] = miny;
    y_max_shared[i] = maxy;

    for (j = bounding_box_threads / 2; j > 0; j /= 2) {
        __syncthreads();
        if (i < j) {
            k = i + j;
            x_min_shared[i] = minx = fminf(minx, x_min_shared[k]);
            x_max_shared[i] = maxx = fmaxf(maxx, x_max_shared[k]);
            y_min_shared[i] = miny = fminf(miny, y_min_shared[k]);
            y_max_shared[i] = maxy = fmaxf(maxy, y_max_shared[k]);
        }
    }

    // write block result to global memory
    if (i == 0) {
        k = blockIdx.x;
        x_min[k] = minx;
        x_max[k] = maxx;
        y_min[k] = miny;
        y_max[k] = maxy;
        __threadfence();

        inc = gridDim.x - 1;
        if (inc == atomicInc(&blkcntd, inc)) {
            // I'm the last block, so combine all block results
            for (j = 0; j <= inc; j++) {
                minx = fminf(minx, x_min_device[j]);
                maxx = fmaxf(maxx, x_max_device[j]);
                miny = fminf(miny, y_min_device[j]);
                maxy = fmaxf(maxy, y_max_device[j]);
            }

            // compute 'radius'
            radiusd = fmaxf(maxx - minx, maxy - miny) * 0.5f + 1e-5f;

            // create root node
            k = N_NODES;
            bottomd = k;

            cell_mass[k] = -1.0f;
            cell_starts[k] = 0;
            embedding_x[k] = (minx + maxx) * 0.5f;
            embedding_y[k] = (miny + maxy) * 0.5f;
            k *= 4;
            for (i = 0; i < 4; i++)
                children[k + i] = -1;

            stepd++;
        }
    }
}



//
namespace BoundingBox_ {


void boundingBox(   int * __restrict__ cell_starts,
                    int * __restrict__ children,
                    float * __restrict__ cell_mass,
                    float * __restrict__ embedding,
                    float * __restrict__ x_max,
                    float * __restrict__ y_max,
                    float * __restrict__ x_min,
                    float * __restrict__ y_min,
                    const int N_NODES,
                    const int n,
                    const in BLOCKS,
                    const int bounding_kernel_factor,
                    const int bounding_kernel_threads)
{
    boundingBoxKernel<<<BLOCKS * bounding_kernel_factor,
                        bounding_kernel_threads,
                        sizeof(float)*4*bounding_kernel_threads>>> \
                    (
                        cell_starts,
                        children,
                        cell_mass,
                        embedding,
                        embedding + N_NODES + 1,
                        x_max,
                        y_max,
                        x_min,
                        y_min,
                        N_NODES, n,
                        bounding_kernel_threads
                    );
    cuda_synchronize();
}


// end namespace
}