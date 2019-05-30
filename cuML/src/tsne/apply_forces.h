
using namespace ML;
#include "utils.h"
#include "cuda_utils.h"


#pragma once

//
namespace ApplyForces_ {


//
__global__
__launch_bounds__(THREADS6, FACTOR6)
void applyForcesKernel(int N,
                        int nnodes,
                        float eta,
                        float norm,
                        float momentum,
                        float exaggeration,
                        volatile float * __restrict__ embedding,
                        volatile float * __restrict__ attraction,
                        volatile float * __restrict__ repulsion,
                        volatile float * __restrict__ gains,
                        volatile float * __restrict__ old_forces)
{
    float dx, dy, ux, uy, gx, gy;

    // iterate over all bodies assigned to thread
    int inc = blockDim.x * gridDim.x;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += inc) {
        ux = old_forces[i];
        uy = old_forces[N + i];
        gx = gains[i];
        gy = gains[N + i];
        dx = exaggeration*attraction[i] - (repulsion[i] / norm);
        dy = exaggeration*attraction[i + N] - (repulsion[nnodes + 1 + i] / norm);

        // Add gains
        gx = (signbit(dx) != signbit(ux)) ? gx + 0.2 : gx * 0.8;
        gy = (signbit(dy) != signbit(uy)) ? gy + 0.2 : gy * 0.8;
        gx = (gx < 0.01) ? 0.01 : gx;
        gy = (gy < 0.01) ? 0.01 : gy;

        // Add momentum
        ux = momentum * ux - eta * gx * dx;
        uy = momentum * uy - eta * gy * dy;

        embedding[i] += ux;
        embedding[i + nnodes + 1] += uy;

        old_forces[i] = ux;
        old_forces[N + i] = uy;
        gains[i] = gx;
        gains[N + i] = gy;
    }
}


// end namespace
}