
using namespace ML;
#include "utils.h"
#include "cuda_utils.h"

#pragma once

namespace Summary_ {

__global__
__launch_bounds__(THREADS3, FACTOR3)
void summarizationKernel(const int N_NODES, 
                        const int nbodiesd, 
                        volatile int * __restrict__ countd, 
                        const int * __restrict__ childd, 
                        volatile float * __restrict__ massd, 
                        volatile float * __restrict__ posxd, 
                        volatile float * __restrict__ posyd) 
{
    int i, j, k, ch, inc, cnt, bottom, flag;
    float m, cm, px, py;
    __shared__ int child[THREADS3 * 4];
    __shared__ float mass[THREADS3 * 4];

    bottom = bottomd;
    inc = blockDim.x * gridDim.x;
    k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;        // align to warp size
    if (k < bottom) k += inc;

    int restart = k;
    int WHERE;
    for (j = 0; j < 5; j++) {        // wait-free pre-passes
        // iterate over all cells assigned to thread
        while (k <= N_NODES) {
            if (massd[k] < 0.0f) {
                for (i = 0; i < 4; i++) {
                    ch = childd[k*4 + i];
                    WHERE = i*THREADS3 + threadIdx.x;
                    child[WHERE] = ch;        // cache children
                    if ((ch >= nbodiesd) && ((mass[WHERE] = massd[ch]) < 0.0f))
                        break;
                }
                if (i == 4) {
                    // all children are ready
                    cm = 0.0f;
                    px = 0.0f;
                    py = 0.0f;
                    cnt = 0;
                    for (i = 0; i < 4; i++) {
                        WHERE = i*THREADS3 + threadIdx.x;
                        ch = child[WHERE];
                        if (ch >= 0) {
                            if (ch >= nbodiesd) {        // count bodies (needed later)
                                m = mass[WHERE];
                                cnt += countd[ch];
                            }
                            else {
                                m = massd[ch];
                                cnt++;
                            }
                            // add child's contribution
                            cm += m;
                            px += posxd[ch] * m;
                            py += posyd[ch] * m;
                        }
                    }
                    countd[k] = cnt;
                    m = 1.0f / cm;
                    posxd[k] = px * m;
                    posyd[k] = py * m;
                    kernel_fence();        // make sure data are visible before setting mass
                    massd[k] = cm;
                }
            }
            k += inc;        // move on to next cell
        }
        k = restart;
    }

    flag = 0;
    j = 0;
    // iterate over all cells assigned to thread
    while (k <= N_NODES) {
        if (massd[k] >= 0.0f) {
            k += inc;
        }
        else {
            if (j == 0) {
                j = 4;
                for (i = 0; i < 4; i++) {
                    ch = childd[k*4 + i];
                    WHERE = i*THREADS3 + threadIdx.x;
                    child[WHERE] = ch;        // cache children
                    if ((ch < nbodiesd) || ((mass[WHERE] = massd[ch]) >= 0.0f))
                        j--;
                }
            }
            else {
                j = 4;
                for (i = 0; i < 4; i++) {
                    WHERE = i*THREADS3 + threadIdx.x;
                    ch = child[WHERE];
                    if ((ch < nbodiesd) || (mass[WHERE] >= 0.0f) || ((mass[WHERE] = massd[ch]) >= 0.0f))
                        j--;
                }
            }

            if (j == 0) {
                // all children are ready
                cm = 0.0f;
                px = 0.0f;
                py = 0.0f;
                cnt = 0;
                for (i = 0; i < 4; i++) {
                    WHERE = i*THREADS3 + threadIdx.x;
                    ch = child[WHERE];
                    if (ch >= 0) {
                        if (ch >= nbodiesd) {        // count bodies (needed later)
                            m = mass[WHERE];
                            cnt += countd[ch];
                        }
                        else {
                            m = massd[ch];
                            cnt++;
                        }
                        // add child's contribution
                        cm += m;
                        px += posxd[ch] * m;
                        py += posyd[ch] * m;
                    }
                }
                countd[k] = cnt;
                m = 1.0f / cm;
                posxd[k] = px * m;
                posyd[k] = py * m;
                flag = 1;
            }
        }
        kernel_sync();        
        // kernel_fence();
        if (flag != 0) {
            massd[k] = cm;
            k += inc;
            flag = 0;
        }
    }
}


//
__global__
__launch_bounds__(THREADS4, FACTOR4)
void sortKernel(const int N_NODES,
                const int nbodiesd,
                int * __restrict__ sortd,
                int * __restrict__ countd,
                volatile int * __restrict__ startd,
                int * __restrict__ childd)
{
    int i, j, k, ch, dec, start, bottom;

    bottom = bottomd;
    dec = blockDim.x * gridDim.x;
    k = N_NODES + 1 - dec + threadIdx.x + blockIdx.x * blockDim.x;

    int k4;
    // iterate over all cells assigned to thread
    while (k >= bottom) {
        start = startd[k];
        if (start >= 0) {
            j = 0;
            for (i = 0; i < 4; i++) {

                k4 = k*4;
                ch = childd[k4 + i];
                if (ch >= 0) {
                    if (i != j) {
                        // move children to front (needed later for speed)
                        childd[k4 + i] = -1;
                        childd[k4 + j] = ch;
                    }
                    j++;
                    if (ch >= nbodiesd) {
                        // child is a cell
                        startd[ch] = start;    // set start ID of child
                        start += countd[ch];    // add #bodies in subtree
                    }
                    else {
                        // child is a body
                        sortd[start] = ch;    // record body in 'sorted' array
                        start++;
                    }
                }
            }
            k -= dec;    // move on to next cell
        }
    }
}

// end namespace
}