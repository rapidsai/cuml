
using namespace ML;
#include "utils.h"
#include "cuda_utils.h"

#pragma once

//
namespace Repulsion_ {


__global__
__launch_bounds__(THREADS5, FACTOR5)
void repulsionKernel(int nnodesd,  int nbodiesd, 
                    volatile int * __restrict__ errd, 
                    float theta, 
                    float epssqd, // correction for zero distance
                    volatile int * __restrict__ sortd, 
                    volatile int * __restrict__ childd, 
                    volatile float * __restrict__ massd, 
                    volatile float * __restrict__ posxd, 
                    volatile float * __restrict__ posyd, 
                    volatile float * __restrict__ velxd, 
                    volatile float * __restrict__ velyd,
                    volatile float * __restrict__ normd) 
{
    int i, j, k, n, depth, base, sbase, diff, pd, nd;
    float px, py, vx, vy, dx, dy, normsum, tmp, mult;
    __shared__ volatile int pos[MAXDEPTH * THREADS5/WARPSIZE], node[MAXDEPTH * THREADS5/WARPSIZE];
    __shared__ float dq[MAXDEPTH * THREADS5/WARPSIZE];

    if (0 == threadIdx.x) {
        dq[0] = (radiusd * radiusd) / (theta * theta); 
        for (i = 1; i < maxdepthd; i++) {
            dq[i] = dq[i - 1] * 0.25f; // radius is halved every level of tree so squared radius is quartered
            dq[i - 1] += epssqd;
        }
        dq[i - 1] += epssqd;

        if (maxdepthd > MAXDEPTH)
            *errd = maxdepthd;
    }
    kernel_sync();


    if (maxdepthd <= MAXDEPTH) {
        // figure out first thread in each warp (lane 0)
        base = threadIdx.x / WARPSIZE;
        sbase = base * WARPSIZE;
        j = base * MAXDEPTH;

        diff = threadIdx.x - sbase;
        // make multiple copies to avoid index calculations later
        if (diff < MAXDEPTH)
            dq[diff+j] = dq[diff];

        kernel_sync();
        __threadfence_block();

        // iterate over all bodies assigned to thread
        for (k = threadIdx.x + blockIdx.x * blockDim.x; k < nbodiesd; k += blockDim.x * gridDim.x) {
            i = sortd[k];    // get permuted/sorted index
            // cache position info
            px = posxd[i];
            py = posyd[i];

            vx = 0.0f;
            vy = 0.0f;
            normsum = 0.0f;

            // initialize iteration stack, i.e., push root node onto stack
            depth = j;
            if (sbase == threadIdx.x) {
                pos[j] = 0;
                node[j] = nnodesd * 4;
            }

            do {
                // stack is not empty
                pd = pos[depth];
                nd = node[depth];
                while (pd < 4) {
                    // node on top of stack has more children to process
                    n = childd[nd + pd];    // load child pointer
                    pd++;

                    if (n >= 0) {
                        dx = px - posxd[n];
                        dy = py - posyd[n];
                        tmp = dx*dx + dy*dy + epssqd; // distance squared plus small constant to prevent zeros
                        #if (CUDART_VERSION >= 9000)
                            if ((n < nbodiesd) || __all_sync(__activemask(), tmp >= dq[depth])) {    // check if all threads agree that cell is far enough away (or is a body)
                        #else
                            if ((n < nbodiesd) || __all(tmp >= dq[depth])) {    // check if all threads agree that cell is far enough away (or is a body)
                        #endif
                            // from bhtsne - sptree.cpp
                            tmp = 1.0f / (1.0f + tmp);
                            mult = massd[n] * tmp;
                            normsum += mult;
                            mult *= tmp;
                            vx += dx * mult;
                            vy += dy * mult;
                        }
                        else {
                            // push cell onto stack
                            if (sbase == threadIdx.x) {    // maybe don't push and inc if last child
                                pos[depth] = pd;
                                node[depth] = nd;
                            }
                            depth++;
                            pd = 0;
                            nd = n * 4;
                        }
                    }
                    else
                        pd = 4;    // early out because all remaining children are also zero
                }
                depth--;    // done with this level
                
            } while (depth >= j);

            if (stepd >= 0) {
                // update velocity
                velxd[i] += vx;
                velyd[i] += vy;
                normd[i] = normsum - 1.0f; // subtract one for self computation (qii)
            }
        }
    }
}

// end namespace
}