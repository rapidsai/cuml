
#pragma once
#include "utils.h"

//
namespace BuildTree_ {

//
__global__
__launch_bounds__(1024, 1)
void clearKernel1(  const int N_NODES,
                    const int nbodiesd,
                    volatile int * __restrict__ childd)
{
    const int top = 4 * N_NODES;
    const int bottom = 4 * nbodiesd;
    const int inc = blockDim.x * gridDim.x;
    int k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;
    if (k < bottom) k += inc;

    // iterate over all cells assigned to thread
    while (k < top) {
        childd[k] = -1;
        k += inc;
    }
}


//
__global__
__launch_bounds__(THREADS2, FACTOR2)
void treeBuildingKernel(const int N_NODES, 
                        const int nbodiesd, 
                        volatile int * __restrict__ errd, 
                        volatile int * __restrict__ childd, 
                        volatile float * __restrict__ posxd, 
                        volatile float * __restrict__ posyd) 
{
    float x, y, r;
    float px, py;
    float dx, dy;
    int ch, n, cell, locked, patch;

    // cache root data
    float radius = radiusd;
    float rootx = posxd[N_NODES];
    float rooty = posyd[N_NODES];

    int localmaxdepth = 1;
    int skip = 1;
    int inc = blockDim.x * gridDim.x;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j;

    // iterate over all bodies assigned to thread
    while (i < nbodiesd) {
        if (skip != 0) {
            // new body, so start traversing at root
            skip = 0;
            px = posxd[i];
            py = posyd[i];
            n = N_NODES;
            depth = 1;
            r = radius * 0.5f;
            dx = dy = -r;
            j = 0;
            // determine which child to follow
            if (rootx < px) {j = 1; dx = r;}
            if (rooty < py) {j |= 2; dy = r;}
            x = rootx + dx;
            y = rooty + dy;
        }

        // follow path to leaf cell
        ch = childd[n*4 + j];
        while (ch >= nbodiesd) {
            n = ch;
            depth++;
            r *= 0.5f;
            dx = dy = -r;
            j = 0;
            // determine which child to follow
            if (x < px) {j = 1; dx = r;}
            if (y < py) {j |= 2; dy = r;}
            x += dx;
            y += dy;
            ch = childd[n*4 + j];
        }

        if (ch != -2) {    // skip if child pointer is locked and try again later
            locked = n*4 + j;
            if (ch == -1) {
                if (-1 == atomicCAS((int *) &childd[locked], -1, i)) {    // if null, just insert the new body
                    localmaxdepth = MAX(depth, localmaxdepth);
                    i += inc;    // move on to next body
                    skip = 1;
                }
            }
            else {    // there already is a body in this position
                if (ch == atomicCAS((int *)&childd[locked], ch, -2)) {    // try to lock
                    patch = -1;
                    // create new cell(s) and insert the old and new body
                    do {
                        depth++;

                        cell = atomicSub((int *)&bottomd, 1) - 1;
                        if (cell <= nbodiesd) {
                            *errd = 1;
                            bottomd = N_NODES;
                        }

                        if (patch != -1)
                            childd[n*4 + j] = cell;

                        patch = MAX(patch, cell);
                        j = 0;
                        if (x < posxd[ch]) j = 1;
                        if (y < posyd[ch]) j |= 2;
                        childd[cell*4 + j] = ch;
                        n = cell;
                        r *= 0.5f;
                        dx = dy = -r;
                        j = 0;
                        if (x < px) {j = 1; dx = r;}
                        if (y < py) {j |= 2; dy = r;}
                        x += dx;
                        y += dy;
                        ch = childd[n*4 + j];
                        // repeat until the two bodies are different children
                    } while (ch >= 0 && r > 1e-10); // add radius check because bodies that are very close together can cause this to fail... there is some error condition here that I'm not entirely sure of (not just when two bodies are equal)
                    
                    childd[n*4 + j] = i;

                    localmaxdepth = MAX(depth, localmaxdepth);
                    i += inc;    // move on to next body
                    skip = 2;
                }
            }
        }
        kernel_fence();

        if (skip == 2)
            childd[locked] = patch;
    }
    // record maximum tree depth
    atomicMax((int *)&maxdepthd, localmaxdepth);
}


//
__global__
__launch_bounds__(1024, 1)
void clearKernel2(  const int N_NODES,
                    volatile int * __restrict__ startd,
                    volatile float * __restrict__ massd)
{   
    const int bottom = bottomd;
    const int inc = blockDim.x * gridDim.x;
    int k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;
    if (k < bottom) k += inc;

    // iterate over all cells assigned to thread
    while (k < N_NODES) {
        massd[k] = -1.0f;
        startd[k] = -1;
        k += inc;
    }
}

// end namespace
}