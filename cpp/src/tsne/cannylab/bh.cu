/*
CUDA BarnesHut v3.1: Simulation of the gravitational forces
in a galactic cluster using the Barnes-Hut n-body algorithm

Copyright (c) 2013, Texas State University-San Marcos. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted for academic, research, experimental, or personal use provided that
the following conditions are met:

   * Redistributions of source code must retain the above copyright notice, 
     this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.
   * Neither the name of Texas State University-San Marcos nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

For all other uses, please contact the Office for Commercialization and Industry
Relations at Texas State University-San Marcos <http://www.txstate.edu/ocir/>.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher <burtscher@txstate.edu>
*/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#include <chrono>
#include "bh_tsne.h"
#include <cuda_runtime_api.h>

#ifndef NO_ZMQ
	#include <zmq.hpp>
#endif

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

// #ifdef __KEPLER__

// #define GPU_ARCH "KEPLER"

// // thread count
// #define THREADS1 1024  /* must be a power of 2 */
// #define THREADS2 1024
// #define THREADS3 768
// #define THREADS4 128
// #define THREADS5 1024
// #define THREADS6 1024

// // block count = factor * #SMs
// #define FACTOR1 2
// #define FACTOR2 2
// #define FACTOR3 1  /* must all be resident at the same time */
// #define FACTOR4 4  /* must all be resident at the same time */
// #define FACTOR5 2
// #define FACTOR6 2

// #elif __MAXWELL__

// #define GPU_ARCH "MAXWELL"

// // thread count
// #define THREADS1 512  /* must be a power of 2 */
// #define THREADS2 512
// #define THREADS3 128
// #define THREADS4 64
// #define THREADS5 256
// #define THREADS6 1024

// // block count = factor * #SMs
// #define FACTOR1 3
// #define FACTOR2 3
// #define FACTOR3 6  /* must all be resident at the same time */
// #define FACTOR4 6  /* must all be resident at the same time */
// #define FACTOR5 5
// #define FACTOR6 1

// #elif __PASCAL__

#define GPU_ARCH "PASCAL"

// thread count
#define THREADS1 512  /* must be a power of 2 */
#define THREADS2 512
#define THREADS3 768
#define THREADS4 128
#define THREADS5 1024
#define THREADS6 1024
#define THREADS7 1024

// block count = factor * #SMs
#define FACTOR1 3
#define FACTOR2 3
#define FACTOR3 1  /* must all be resident at the same time */
#define FACTOR4 4  /* must all be resident at the same time */
#define FACTOR5 2
#define FACTOR6 2
#define FACTOR7 1

// #else

// #define GPU_ARCH "UNKNOWN"

// // thread count
// #define THREADS1 512  /* must be a power of 2 */
// #define THREADS2 512
// #define THREADS3 128
// #define THREADS4 64
// #define THREADS5 256
// #define THREADS6 1024

// // block count = factor * #SMs
// #define FACTOR1 3
// #define FACTOR2 3
// #define FACTOR3 6  /* must all be resident at the same time */
// #define FACTOR4 6  /* must all be resident at the same time */
// #define FACTOR5 5
// #define FACTOR6 1

// #endif

#define WARPSIZE 32
#define MAXDEPTH 32

__device__ volatile int stepd, bottomd, maxdepthd;
__device__ unsigned int blkcntd;
__device__ volatile float radiusd;

/******************************************************************************/
/*** initialize memory ********************************************************/
/******************************************************************************/

__global__ void InitializationKernel(int * __restrict errd)
{
  *errd = 0;
  stepd = -1;
  maxdepthd = 1;
  blkcntd = 0;
}


/******************************************************************************/
/*** compute center and radius ************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS1, FACTOR1)
void BoundingBoxKernel(int nnodesd, 
                        int nbodiesd, 
                        volatile int * __restrict startd, 
                        volatile int * __restrict childd, 
                        volatile float * __restrict massd, 
                        volatile float * __restrict posxd, 
                        volatile float * __restrict posyd, 
                        volatile float * __restrict maxxd, 
                        volatile float * __restrict maxyd, 
                        volatile float * __restrict minxd, 
                        volatile float * __restrict minyd) 
{
  register int i, j, k, inc;
  register float val, minx, maxx, miny, maxy;
  __shared__ volatile float sminx[THREADS1], smaxx[THREADS1], sminy[THREADS1], smaxy[THREADS1];

  // initialize with valid data (in case #bodies < #threads)
  minx = maxx = posxd[0];
  miny = maxy = posyd[0];

  // scan all bodies
  i = threadIdx.x;
  inc = THREADS1 * gridDim.x;
  for (j = i + blockIdx.x * THREADS1; j < nbodiesd; j += inc) {
    val = posxd[j];
    minx = fminf(minx, val);
    maxx = fmaxf(maxx, val);
    val = posyd[j];
    miny = fminf(miny, val);
    maxy = fmaxf(maxy, val);
  }

  // reduction in shared memory
  sminx[i] = minx;
  smaxx[i] = maxx;
  sminy[i] = miny;
  smaxy[i] = maxy;

  for (j = THREADS1 / 2; j > 0; j /= 2) {
    __syncthreads();
    if (i < j) {
      k = i + j;
      sminx[i] = minx = fminf(minx, sminx[k]);
      smaxx[i] = maxx = fmaxf(maxx, smaxx[k]);
      sminy[i] = miny = fminf(miny, sminy[k]);
      smaxy[i] = maxy = fmaxf(maxy, smaxy[k]);
    }
  }

  // write block result to global memory
  if (i == 0) {
    k = blockIdx.x;
    minxd[k] = minx;
    maxxd[k] = maxx;
    minyd[k] = miny;
    maxyd[k] = maxy;
    __threadfence();

    inc = gridDim.x - 1;
    if (inc == atomicInc(&blkcntd, inc)) {
      // I'm the last block, so combine all block results
      for (j = 0; j <= inc; j++) {
        minx = fminf(minx, minxd[j]);
        maxx = fmaxf(maxx, maxxd[j]);
        miny = fminf(miny, minyd[j]);
        maxy = fmaxf(maxy, maxyd[j]);
      }

      // compute 'radius'
      radiusd = fmaxf(maxx - minx, maxy - miny) * 0.5f + 1e-5f;

      // create root node
      k = nnodesd;
      bottomd = k;

      massd[k] = -1.0f;
      startd[k] = 0;
      posxd[k] = (minx + maxx) * 0.5f;
      posyd[k] = (miny + maxy) * 0.5f;
      k *= 4;
      for (i = 0; i < 4; i++) childd[k + i] = -1;

      stepd++;
    }
  }
}


/******************************************************************************/
/*** build tree ***************************************************************/
/******************************************************************************/

__global__
__launch_bounds__(1024, 1)
void ClearKernel1(int nnodesd, int nbodiesd, volatile int * __restrict childd)
{
  register int k, inc, top, bottom;

  top = 4 * nnodesd;
  bottom = 4 * nbodiesd;
  inc = blockDim.x * gridDim.x;
  k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;  // align to warp size
  if (k < bottom) k += inc;

  // iterate over all cells assigned to thread
  while (k < top) {
    childd[k] = -1;
    k += inc;
  }
}


__global__
__launch_bounds__(THREADS2, FACTOR2)
void TreeBuildingKernel(int nnodesd, 
                        int nbodiesd, 
                        volatile int * __restrict errd, 
                        volatile int * __restrict childd, 
                        volatile float * __restrict posxd, 
                        volatile float * __restrict posyd) 
{
  register int i, j, depth, localmaxdepth, skip, inc;
  register float x, y, r;
  register float px, py;
  register float dx, dy;
  register int ch, n, cell, locked, patch;
  register float radius, rootx, rooty;

  // cache root data
  radius = radiusd;
  rootx = posxd[nnodesd];
  rooty = posyd[nnodesd];

  localmaxdepth = 1;
  skip = 1;
  inc = blockDim.x * gridDim.x;
  i = threadIdx.x + blockIdx.x * blockDim.x;

  // iterate over all bodies assigned to thread
  while (i < nbodiesd) {
    //   if (TID == 0)
        // printf("\tStarting\n");
    if (skip != 0) {
      // new body, so start traversing at root
      skip = 0;
      px = posxd[i];
      py = posyd[i];
      n = nnodesd;
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
    ch = childd[n*4+j];
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
      ch = childd[n*4+j];
    }
    if (ch != -2) {  // skip if child pointer is locked and try again later
      locked = n*4+j;
      if (ch == -1) {
        if (-1 == atomicCAS((int *)&childd[locked], -1, i)) {  // if null, just insert the new body
          localmaxdepth = max(depth, localmaxdepth);
          i += inc;  // move on to next body
          skip = 1;
        }
      } else {  // there already is a body in this position
        if (ch == atomicCAS((int *)&childd[locked], ch, -2)) {  // try to lock
          patch = -1;
          // create new cell(s) and insert the old and new body
          do {
            depth++;

            cell = atomicSub((int *)&bottomd, 1) - 1;
            if (cell <= nbodiesd) {
              *errd = 1;
              bottomd = nnodesd;
            }

            if (patch != -1) {
              childd[n*4+j] = cell;
            }
            patch = max(patch, cell);
            j = 0;
            if (x < posxd[ch]) j = 1;
            if (y < posyd[ch]) j |= 2;
            childd[cell*4+j] = ch;
            n = cell;
            r *= 0.5f;
            dx = dy = -r;
            j = 0;
            if (x < px) {j = 1; dx = r;}
            if (y < py) {j |= 2; dy = r;}
            x += dx;
            y += dy;
            ch = childd[n*4+j];
            // repeat until the two bodies are different children
          } while (ch >= 0 && r > 1e-10); // add radius check because bodies that are very close together can cause this to fail... there is some error condition here that I'm not entirely sure of (not just when two bodies are equal)
          childd[n*4+j] = i;

          localmaxdepth = max(depth, localmaxdepth);
          i += inc;  // move on to next body
          skip = 2;
        }
      }
    }
    __threadfence();

    if (skip == 2) {
      childd[locked] = patch;
    }
  }
  // record maximum tree depth
  atomicMax((int *)&maxdepthd, localmaxdepth);
}


__global__
__launch_bounds__(1024, 1)
void ClearKernel2(int nnodesd, volatile int * __restrict startd, volatile float * __restrict massd)
{
  register int k, inc, bottom;

  bottom = bottomd;
  inc = blockDim.x * gridDim.x;
  k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;  // align to warp size
  if (k < bottom) k += inc;

  // iterate over all cells assigned to thread
  while (k < nnodesd) {
    massd[k] = -1.0f;
    startd[k] = -1;
    k += inc;
  }
}


/******************************************************************************/
/*** compute center of mass ***************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS3, FACTOR3)
void SummarizationKernel(const int nnodesd, 
                            const int nbodiesd, 
                            volatile int * __restrict countd, 
                            const int * __restrict childd, 
                            volatile float * __restrict massd, 
                            volatile float * __restrict posxd, 
                            volatile float * __restrict posyd) 
{
  register int i, j, k, ch, inc, cnt, bottom, flag;
  register float m, cm, px, py;
  __shared__ int child[THREADS3 * 4];
  __shared__ float mass[THREADS3 * 4];

  bottom = bottomd;
  inc = blockDim.x * gridDim.x;
  k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;  // align to warp size
  if (k < bottom) k += inc;

  register int restart = k;
  for (j = 0; j < 5; j++) {  // wait-free pre-passes
    // iterate over all cells assigned to thread
    while (k <= nnodesd) {
      if (massd[k] < 0.0f) {
        for (i = 0; i < 4; i++) {
          ch = childd[k*4+i];
          child[i*THREADS3+threadIdx.x] = ch;  // cache children
          if ((ch >= nbodiesd) && ((mass[i*THREADS3+threadIdx.x] = massd[ch]) < 0.0f)) {
            break;
          }
        }
        if (i == 4) {
          // all children are ready
          cm = 0.0f;
          px = 0.0f;
          py = 0.0f;
          cnt = 0;
          for (i = 0; i < 4; i++) {
            ch = child[i*THREADS3+threadIdx.x];
            if (ch >= 0) {
              if (ch >= nbodiesd) {  // count bodies (needed later)
                m = mass[i*THREADS3+threadIdx.x];
                cnt += countd[ch];
              } else {
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
          __threadfence();  // make sure data are visible before setting mass
          massd[k] = cm;
        }
      }
      k += inc;  // move on to next cell
    }
    k = restart;
  }

  flag = 0;
  j = 0;
  // iterate over all cells assigned to thread
  while (k <= nnodesd) {
    if (massd[k] >= 0.0f) {
      k += inc;
    } else {
      if (j == 0) {
        j = 4;
        for (i = 0; i < 4; i++) {
          ch = childd[k*4+i];
          child[i*THREADS3+threadIdx.x] = ch;  // cache children
          if ((ch < nbodiesd) || ((mass[i*THREADS3+threadIdx.x] = massd[ch]) >= 0.0f)) {
            j--;
          }
        }
      } else {
        j = 4;
        for (i = 0; i < 4; i++) {
          ch = child[i*THREADS3+threadIdx.x];
          if ((ch < nbodiesd) || (mass[i*THREADS3+threadIdx.x] >= 0.0f) || ((mass[i*THREADS3+threadIdx.x] = massd[ch]) >= 0.0f)) {
            j--;
          }
        }
      }

      if (j == 0) {
        // all children are ready
        cm = 0.0f;
        px = 0.0f;
        py = 0.0f;
        cnt = 0;
        for (i = 0; i < 4; i++) {
          ch = child[i*THREADS3+threadIdx.x];
          if (ch >= 0) {
            if (ch >= nbodiesd) {  // count bodies (needed later)
              m = mass[i*THREADS3+threadIdx.x];
              cnt += countd[ch];
            } else {
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
    __syncthreads();  
    // __threadfence();
    if (flag != 0) {
      massd[k] = cm;
      k += inc;
      flag = 0;
    }
  }
}


/******************************************************************************/
/*** sort bodies **************************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS4, FACTOR4)
void SortKernel(int nnodesd, int nbodiesd, int * __restrict sortd, int * __restrict countd, volatile int * __restrict startd, int * __restrict childd)
{
  register int i, j, k, ch, dec, start, bottom;

  bottom = bottomd;
  dec = blockDim.x * gridDim.x;
  k = nnodesd + 1 - dec + threadIdx.x + blockIdx.x * blockDim.x;

  // iterate over all cells assigned to thread
  while (k >= bottom) {
    start = startd[k];
    if (start >= 0) {
      j = 0;
      for (i = 0; i < 4; i++) {
        ch = childd[k*4+i];
        if (ch >= 0) {
          if (i != j) {
            // move children to front (needed later for speed)
            childd[k*4+i] = -1;
            childd[k*4+j] = ch;
          }
          j++;
          if (ch >= nbodiesd) {
            // child is a cell
            startd[ch] = start;  // set start ID of child
            start += countd[ch];  // add #bodies in subtree
          } else {
            // child is a body
            sortd[start] = ch;  // record body in 'sorted' array
            start++;
          }
        }
      }
      k -= dec;  // move on to next cell
    }
  }
}


/******************************************************************************/
/*** compute force ************************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS5, FACTOR5)
void ForceCalculationKernel(int nnodesd, 
                            int nbodiesd, 
                            volatile int * __restrict errd, 
                            float theta, 
                            float epssqd, // correction for zero distance
                            volatile int * __restrict sortd, 
                            volatile int * __restrict childd, 
                            volatile float * __restrict massd, 
                            volatile float * __restrict posxd, 
                            volatile float * __restrict posyd, 
                            volatile float * __restrict velxd, 
                            volatile float * __restrict velyd,
                            volatile float * __restrict normd) 
{
  register int i, j, k, n, depth, base, sbase, diff, pd, nd;
  register float px, py, vx, vy, dx, dy, normsum, tmp, mult;
  __shared__ volatile int pos[MAXDEPTH * THREADS5/WARPSIZE], node[MAXDEPTH * THREADS5/WARPSIZE];
  __shared__ float dq[MAXDEPTH * THREADS5/WARPSIZE];

  if (0 == threadIdx.x) {
    dq[0] = (radiusd * radiusd) / (theta * theta); 
    for (i = 1; i < maxdepthd; i++) {
        dq[i] = dq[i - 1] * 0.25f; // radius is halved every level of tree so squared radius is quartered
        dq[i - 1] += epssqd;
    }
    dq[i - 1] += epssqd;

    if (maxdepthd > MAXDEPTH) {
      *errd = maxdepthd;
    }
  }
  __syncthreads();

  if (maxdepthd <= MAXDEPTH) {
    // figure out first thread in each warp (lane 0)
    base = threadIdx.x / WARPSIZE;
    sbase = base * WARPSIZE;
    j = base * MAXDEPTH;

    diff = threadIdx.x - sbase;
    // make multiple copies to avoid index calculations later
    if (diff < MAXDEPTH) {
      dq[diff+j] = dq[diff];
    }
    __syncthreads();
    __threadfence_block();

    // iterate over all bodies assigned to thread
    for (k = threadIdx.x + blockIdx.x * blockDim.x; k < nbodiesd; k += blockDim.x * gridDim.x) {
      i = sortd[k];  // get permuted/sorted index
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
          n = childd[nd + pd];  // load child pointer
          pd++;

          if (n >= 0) {
            dx = px - posxd[n];
            dy = py - posyd[n];
            tmp = dx*dx + dy*dy + epssqd; // distance squared plus small constant to prevent zeros
            #if (CUDART_VERSION >= 9000)
              if ((n < nbodiesd) || __all_sync(__activemask(), tmp >= dq[depth])) {  // check if all threads agree that cell is far enough away (or is a body)
            #else
              if ((n < nbodiesd) || __all(tmp >= dq[depth])) {  // check if all threads agree that cell is far enough away (or is a body)
            #endif
              // from bhtsne - sptree.cpp
              tmp = 1 / (1 + tmp);
              mult = massd[n] * tmp;
              normsum += mult;
              mult *= tmp;
              vx += dx * mult;
              vy += dy * mult;
            } else {
              // push cell onto stack
              if (sbase == threadIdx.x) {  // maybe don't push and inc if last child
                pos[depth] = pd;
                node[depth] = nd;
              }
              depth++;
              pd = 0;
              nd = n * 4;
            }
          } else {
            pd = 4;  // early out because all remaining children are also zero
          }
        }
        depth--;  // done with this level
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


/******************************************************************************/
/*** advance bodies ***********************************************************/
/******************************************************************************/
// Edited to add momentum, repulsive, attr forces, etc.
__global__
__launch_bounds__(THREADS6, FACTOR6)
void IntegrationKernel(int N,
                        int nnodes,
                        float eta,
                        float norm,
                        float momentum,
                        float exaggeration,
                        volatile float * __restrict pts, // (nnodes + 1) x 2
                        volatile float * __restrict attr_forces, // (N x 2)
                        volatile float * __restrict rep_forces, // (nnodes + 1) x 2
                        volatile float * __restrict gains,
                        volatile float * __restrict old_forces) // (N x 2)
{
  register int i, inc;
  register float dx, dy, ux, uy, gx, gy;

  // iterate over all bodies assigned to thread
  inc = blockDim.x * gridDim.x;
  for (i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += inc) {
        ux = old_forces[i];
        uy = old_forces[N + i];
        gx = gains[i];
        gy = gains[N + i];
        dx = exaggeration*attr_forces[i] - (rep_forces[i] / norm);
        dy = exaggeration*attr_forces[i + N] - (rep_forces[nnodes + 1 + i] / norm);

        gx = (signbit(dx) != signbit(ux)) ? gx + 0.2 : gx * 0.8;
        gy = (signbit(dy) != signbit(uy)) ? gy + 0.2 : gy * 0.8;
        gx = (gx < 0.01) ? 0.01 : gx;
        gy = (gy < 0.01) ? 0.01 : gy;

        ux = momentum * ux - eta * gx * dx;
        uy = momentum * uy - eta * gy * dy;

        pts[i] += ux;
        pts[i + nnodes + 1] += uy;

        old_forces[i] = ux;
        old_forces[N + i] = uy;
        gains[i] = gx;
        gains[N + i] = gy;
   }
}


/******************************************************************************/
/*** compute attractive force *************************************************/
/******************************************************************************/
__global__
void csr2coo(int N, int nnz, 
                    volatile int   * __restrict pijRowPtr,
                    volatile int   * __restrict pijColInd,
                    volatile int   * __restrict indices)
{
    register int TID, i, j, start, end;
    TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= nnz) return;
    start = 0; end = N + 1;
    i = (N + 1) >> 1;
    while (end - start > 1) {
      j = pijRowPtr[i];
      end = (j <= TID) ? end : i;
      start = (j > TID) ? start : i;
      i = (start + end) >> 1;
    }
    j = pijColInd[TID];
    indices[2*TID] = i;
    indices[2*TID+1] = j;
}

__global__
void ComputePijKernel(const unsigned int N,
                      const unsigned int K,
                      float * __restrict pij,
                      const float * __restrict sqdist,
                      const float * __restrict betas)
{
    register int TID, i, j;
    register float dist, beta;

    TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= N * K) return;
    i = TID / K;
    j = TID % K;

    beta = betas[i];
    dist = sqdist[TID];
    pij[TID] = (j == 0 && dist == 0.0f) ? 0.0f : __expf(-beta * dist); // condition deals with evaluation of pii
}

__global__
void ComputePijxQijKernel(int N, int nnz, int nnodes,
                    volatile int   * indices,
                    volatile float * __restrict pij,
                    volatile float * __restrict forceProd,
                    volatile float * __restrict pts)
{
    register int TID, i, j; //, inc;
    register float ix, iy, jx, jy, dx, dy;
    TID = threadIdx.x + blockIdx.x * blockDim.x;
    // inc = blockDim.x * gridDim.x;
    // for (TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nnz; TID += inc) {
      if (TID >= nnz) return;
      i = indices[2*TID];
      j = indices[2*TID+1];
      ix = pts[i]; iy = pts[nnodes + 1 + i];
      jx = pts[j]; jy = pts[nnodes + 1 + j];
      dx = ix - jx;
      dy = iy - jy;
      forceProd[TID] = pij[TID] * 1 / (1 + dx*dx + dy*dy);
    // }
}

__global__
void PerplexitySearchKernel(const unsigned int N,
                            const float perplexity_target,
                            const float eps,
                            float * __restrict betas,
                            float * __restrict lower_bound,
                            float * __restrict upper_bound,
                            int   * __restrict found,
                            const float * __restrict neg_entropy,
                            const float * __restrict row_sum) 
{
    register int i, is_found;
    register float perplexity, neg_ent, sum_P, pdiff, beta, min_beta, max_beta;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N) return;

    neg_ent = neg_entropy[i];
    sum_P = row_sum[i];
    beta = betas[i];

    min_beta = lower_bound[i];
    max_beta = upper_bound[i];

    perplexity = (neg_ent / sum_P) + __logf(sum_P);
    pdiff = perplexity - __logf(perplexity_target);
    is_found = (pdiff < eps && - pdiff < eps);
    if (!is_found) {
        if (pdiff > 0) {
            min_beta = beta;
            beta = (max_beta == FLT_MAX || max_beta == -FLT_MAX) ? beta * 2.0f : (beta + max_beta) / 2.0f;
        } else {
            max_beta = beta;
            beta = (min_beta == -FLT_MAX || min_beta == FLT_MAX) ? beta / 2.0f : (beta + min_beta) / 2.0f;
        }
        lower_bound[i] = min_beta;
        upper_bound[i] = max_beta;
        betas[i] = beta;
    }
    found[i] = is_found;
}
// computes unnormalized attractive forces
void computeAttrForce(int N,
                        int nnz,
                        int nnodes,
                        int attr_forces_grid_size,
                        int attr_forces_block_size,
                        cusparseHandle_t &handle,
                        cusparseMatDescr_t &descr,
                        thrust::device_vector<float> &sparsePij,
                        thrust::device_vector<int>   &pijRowPtr, // (N + 1)-D vector, should be constant L
                        thrust::device_vector<int>   &pijColInd, // NxL matrix (same shape as sparsePij)
                        thrust::device_vector<float> &forceProd, // NxL matrix
                        thrust::device_vector<float> &pts,       // (nnodes + 1) x 2 matrix
                        thrust::device_vector<float> &forces,    // N x 2 matrix
                        thrust::device_vector<float> &ones,
                        thrust::device_vector<int> &indices)      // N x 2 matrix of ones
{
    // Computes pij*qij for each i,j
    ComputePijxQijKernel<<<attr_forces_grid_size,attr_forces_block_size>>>(N, nnz, nnodes,
                                        thrust::raw_pointer_cast(indices.data()),
                                        thrust::raw_pointer_cast(sparsePij.data()),
                                        thrust::raw_pointer_cast(forceProd.data()),
                                        thrust::raw_pointer_cast(pts.data()));
    // ComputePijxQijKernel<<<blocks*FACTOR7,THREADS7>>>(N, nnz, nnodes,
    //                                     thrust::raw_pointer_cast(indices.data()),
    //                                     thrust::raw_pointer_cast(sparsePij.data()),
    //                                     thrust::raw_pointer_cast(forceProd.data()),
    //                                     thrust::raw_pointer_cast(pts.data()));
    GpuErrorCheck(cudaDeviceSynchronize());

    // compute forces_i = sum_j pij*qij*normalization*yi
    float alpha = 1.0f;
    float beta = 0.0f;
    CusparseSafeCall(cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            N, 2, N, nnz, &alpha, descr,
                            thrust::raw_pointer_cast(forceProd.data()),
                            thrust::raw_pointer_cast(pijRowPtr.data()),
                            thrust::raw_pointer_cast(pijColInd.data()),
                            thrust::raw_pointer_cast(ones.data()),
                            N, &beta, thrust::raw_pointer_cast(forces.data()),
                            N));
    GpuErrorCheck(cudaDeviceSynchronize());
    thrust::transform(forces.begin(), forces.begin() + N, pts.begin(), forces.begin(), thrust::multiplies<float>());
    thrust::transform(forces.begin() + N, forces.end(), pts.begin() + nnodes + 1, forces.begin() + N, thrust::multiplies<float>());

    // compute forces_i = forces_i - sum_j pij*qij*normalization*yj
    alpha = -1.0f;
    beta = 1.0f;
    CusparseSafeCall(cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            N, 2, N, nnz, &alpha, descr,
                            thrust::raw_pointer_cast(forceProd.data()),
                            thrust::raw_pointer_cast(pijRowPtr.data()),
                            thrust::raw_pointer_cast(pijColInd.data()),
                            thrust::raw_pointer_cast(pts.data()),
                            nnodes + 1, &beta, thrust::raw_pointer_cast(forces.data()),
                            N));
    GpuErrorCheck(cudaDeviceSynchronize());
    

}

// TODO: Add -1 notification here... and how to deal with it if it happens
// TODO: Maybe think about getting FAISS to return integers (long-term todo)
__global__ void postprocess_matrix(float* matrix, 
                                    long* long_indices,
                                    int* indices,
                                    unsigned int N_POINTS,
                                    unsigned int K) 
{
    register int TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= N_POINTS*K) return;

    // Set pij to 0 for each of the broken values - Note: this should be handled in the ComputePijKernel now
    // if (matrix[TID] == 1.0f) matrix[TID] = 0.0f;
    indices[TID] = (int) long_indices[TID];
    return;
}

thrust::device_vector<float> search_perplexity(cublasHandle_t &handle,
                                                  thrust::device_vector<float> &knn_distances,
                                                  const float perplexity_target,
                                                  const float eps,
                                                  const unsigned int N,
                                                  const unsigned int K) 
{
    // use beta instead of sigma (this matches the bhtsne code but not the paper)
    // beta is just multiplicative instead of divisive (changes the way binary search works)
    thrust::device_vector<float> betas(N, 1.0f);
    thrust::device_vector<float> lbs(N, 0.0f);
    thrust::device_vector<float> ubs(N, 1000.0f);
    thrust::device_vector<float> pij(N*K);
    thrust::device_vector<float> entropy(N*K);
    thrust::device_vector<int> found(N);

    const unsigned int BLOCKSIZE1 = 1024;
    const unsigned int NBLOCKS1 = iDivUp(N * K, BLOCKSIZE1);

    const unsigned int BLOCKSIZE2 = 128;
    const unsigned int NBLOCKS2 = iDivUp(N, BLOCKSIZE2);

    int iters = 0;
    int all_found = 0;
    thrust::device_vector<float> row_sum;
    do {
        // compute Gaussian Kernel row
        ComputePijKernel<<<NBLOCKS1, BLOCKSIZE1>>>(N, K, thrust::raw_pointer_cast(pij.data()),
                                                            thrust::raw_pointer_cast(knn_distances.data()),
                                                            thrust::raw_pointer_cast(betas.data()));
        GpuErrorCheck(cudaDeviceSynchronize());
        
        // compute entropy of current row
        row_sum = tsnecuda::util::ReduceSum(handle, pij, K, N, 0);
        thrust::transform(pij.begin(), pij.end(), entropy.begin(), tsnecuda::util::FunctionalEntropy());
        auto neg_entropy = tsnecuda::util::ReduceAlpha(handle, entropy, K, N, -1.0f, 0);

        // binary search for beta
        PerplexitySearchKernel<<<NBLOCKS2, BLOCKSIZE2>>>(N, perplexity_target, eps,
                                                            thrust::raw_pointer_cast(betas.data()),
                                                            thrust::raw_pointer_cast(lbs.data()),
                                                            thrust::raw_pointer_cast(ubs.data()),
                                                            thrust::raw_pointer_cast(found.data()),
                                                            thrust::raw_pointer_cast(neg_entropy.data()),
                                                            thrust::raw_pointer_cast(row_sum.data()));
        GpuErrorCheck(cudaDeviceSynchronize());
        all_found = thrust::reduce(found.begin(), found.end(), 1, thrust::minimum<int>());
        iters++;
    } while (!all_found && iters < 200);
    // TODO: Warn if iters == 200 because perplexity not found?

    tsnecuda::util::BroadcastMatrixVector(pij, row_sum, K, N, thrust::divides<float>(), 1, 1.0f);
    return pij;
}

void BHTSNE::tsne(cublasHandle_t &dense_handle, cusparseHandle_t &sparse_handle, BHTSNE::Options &opt) {

    // Check the validity of the options file
    if (!opt.validate()) {
        std::cout << "E: Invalid options file. Terminating." << std::endl;
        return;
    }

    // Setup some return information if we're working on snapshots
    int snap_interval;
    int snap_num = 0;
    if (opt.return_style == BHTSNE::RETURN_STYLE::SNAPSHOT) {
      snap_interval = opt.iterations / (opt.num_snapshots-1);
    }

    // Setup clock information
    auto end_time = std::chrono::high_resolution_clock::now();
    auto start_time = std::chrono::high_resolution_clock::now();
    int times[30]; for (int i = 0; i < 30; i++) times[i] = 0;

    // Allocate Memory for the KNN problem and do some configuration
    start_time = std::chrono::high_resolution_clock::now();

        // Setup CUDA configs
        cudaFuncSetCacheConfig(BoundingBoxKernel, cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(TreeBuildingKernel, cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(ClearKernel1, cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(ClearKernel2, cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(SummarizationKernel, cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(SortKernel, cudaFuncCachePreferL1);
        #ifdef __KEPLER__
        cudaFuncSetCacheConfig(ForceCalculationKernel, cudaFuncCachePreferEqual);
        #else
        cudaFuncSetCacheConfig(ForceCalculationKernel, cudaFuncCachePreferL1);
        #endif
        cudaFuncSetCacheConfig(IntegrationKernel, cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(ComputePijxQijKernel, cudaFuncCachePreferShared);
    
        // Allocate some memory
        const unsigned int K = opt.n_neighbors < opt.n_points ? opt.n_neighbors : opt.n_points - 1; 
        float *knn_distances = new float[opt.n_points*K];
        memset(knn_distances, 0, opt.n_points * K * sizeof(float));
        long *knn_indices = new long[opt.n_points*K]; // Allocate memory for the indices on the CPU
    
    end_time = std::chrono::high_resolution_clock::now();
    times[0] = std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time).count();

    // Compute the KNNs and distances
    start_time = std::chrono::high_resolution_clock::now();

        // Do KNN Call
        tsnecuda::util::KNearestNeighbors(knn_indices, knn_distances, opt.points, opt.n_dims, opt.n_points, K);
        
    end_time = std::chrono::high_resolution_clock::now();
    times[1] = std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time).count();

    // Copy the distances to the GPU and compute Pij
    start_time = std::chrono::high_resolution_clock::now();

        // Allocate device distance memory
        thrust::device_vector<float> d_knn_distances(knn_distances, knn_distances + (opt.n_points * K));
        tsnecuda::util::MaxNormalizeDeviceVector(d_knn_distances); // Here, the extra 0s floating around won't matter
        thrust::device_vector<float> d_pij = search_perplexity(dense_handle, d_knn_distances, opt.perplexity, opt.perplexity_search_epsilon, opt.n_points, K);

        // Clean up distance memory
        d_knn_distances.clear();
        d_knn_distances.shrink_to_fit();

        // Copy the distances back to the GPU
        thrust::device_vector<long> d_knn_indices_long(knn_indices, knn_indices + opt.n_points*K);
        thrust::device_vector<int> d_knn_indices(opt.n_points*K);

        // Post-process the floating point matrix
        const int NBLOCKS_PP = iDivUp(opt.n_points*K, 128);
        postprocess_matrix<<< NBLOCKS_PP, 128 >>>(thrust::raw_pointer_cast(d_pij.data()), 
                                                thrust::raw_pointer_cast(d_knn_indices_long.data()), 
                                                thrust::raw_pointer_cast(d_knn_indices.data()),  opt.n_points, K);
        cudaDeviceSynchronize();

        // Clean up extra memory
        d_knn_indices_long.clear();
        d_knn_indices_long.shrink_to_fit();
        delete[] knn_distances;
        delete[] knn_indices;

    end_time = std::chrono::high_resolution_clock::now();
    times[2] = std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time).count();

    // Symmetrize the Pij matrix
    start_time = std::chrono::high_resolution_clock::now();

        // Construct sparse matrix descriptor
        cusparseMatDescr_t descr;
        cusparseCreateMatDescr(&descr);
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

        // Compute the symmetrized matrix
        thrust::device_vector<float> sparsePij; // Device
        thrust::device_vector<int> pijRowPtr; // Device
        thrust::device_vector<int> pijColInd; // Device
        tsnecuda::util::SymmetrizeMatrix(sparse_handle, 
          d_pij, d_knn_indices, sparsePij, pijColInd, pijRowPtr, opt.n_points, K, opt.magnitude_factor);

        // Clear some old memory
        d_knn_indices.clear();
        d_knn_indices.shrink_to_fit();
        d_pij.clear();
        d_pij.shrink_to_fit(); 

    end_time = std::chrono::high_resolution_clock::now();
    times[3] = std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time).count();


    // Do setup for Barnes-Hut computation
    start_time = std::chrono::high_resolution_clock::now();

        // Compute the CUDA device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        if (deviceProp.warpSize != WARPSIZE) {
        fprintf(stderr, "Warp size must be %d\n", deviceProp.warpSize);
        exit(-1);
        }
        int blocks = deviceProp.multiProcessorCount;
        std::cout << "Multiprocessor Count: " << blocks << std::endl;
        std::cout << "GPU Architecture: " << GPU_ARCH << std::endl;

        // Figure out the number of nodes needed for the BH tree
        int nnodes = opt.n_points * 2;
        if (nnodes < 1024*blocks) nnodes = 1024*blocks;
        while ((nnodes & (WARPSIZE-1)) != 0) nnodes++;
        nnodes--;

        opt.n_nodes = nnodes;

        std::cout << "Number of nodes chosen: " << nnodes << std::endl;

        int attr_forces_block_size;
        int attr_forces_min_grid_size;
        int attr_forces_grid_size;
        cudaOccupancyMaxPotentialBlockSize( &attr_forces_min_grid_size, &attr_forces_block_size, ComputePijxQijKernel, 0, 0);
        attr_forces_grid_size = (sparsePij.size() + attr_forces_block_size - 1) / attr_forces_block_size;
        std::cout << "Autotuned attractive force kernel - Grid size: " << attr_forces_grid_size << " Block Size: " << attr_forces_block_size << std::endl;
        

        // Allocate memory for the barnes hut implementations
        thrust::device_vector<float> forceProd(sparsePij.size());
        thrust::device_vector<float> rep_forces((nnodes + 1) * 2, 0);
        thrust::device_vector<float> attr_forces(opt.n_points * 2, 0);
        thrust::device_vector<float> gains(opt.n_points * 2, 1);
        thrust::device_vector<float> old_forces(opt.n_points * 2, 0); // for momentum
        thrust::device_vector<int> errl(1);
        thrust::device_vector<int> startl(nnodes + 1);
        thrust::device_vector<int> childl((nnodes + 1) * 4);
        thrust::device_vector<float> massl(nnodes + 1, 1.0); // TODO: probably don't need massl
        thrust::device_vector<int> countl(nnodes + 1);
        thrust::device_vector<int> sortl(nnodes + 1);
        thrust::device_vector<float> norml(nnodes + 1);
        thrust::device_vector<float> maxxl(blocks * FACTOR1);
        thrust::device_vector<float> maxyl(blocks * FACTOR1);
        thrust::device_vector<float> minxl(blocks * FACTOR1);
        thrust::device_vector<float> minyl(blocks * FACTOR1);
        thrust::device_vector<float> ones(opt.n_points * 2, 1); // This is for reduce summing, etc.
        thrust::device_vector<int> indices(sparsePij.size()*2);

        // Compute the indices setup
        const int SBS = 1024;
        const int NBS = iDivUp(sparsePij.size(), SBS);
        csr2coo<<<NBS, SBS>>>(opt.n_points, sparsePij.size(), 
                                        thrust::raw_pointer_cast(pijRowPtr.data()),
                                        thrust::raw_pointer_cast(pijColInd.data()),
                                        thrust::raw_pointer_cast(indices.data()));
        GpuErrorCheck(cudaDeviceSynchronize());

        // Point initialization
        thrust::device_vector<float> pts((nnodes + 1) * 2);
        thrust::device_vector<float> random_vec(pts.size());
        
        if (opt.initialization == BHTSNE::TSNE_INIT::UNIFORM) { // Random uniform initialization
            pts = tsnecuda::util::RandomDeviceVectorInRange((nnodes+1)*2, -100, 100);
        } else if (opt.initialization == BHTSNE::TSNE_INIT::GAUSSIAN) { // Random gaussian initialization
            std::default_random_engine generator;
            std::normal_distribution<double> distribution1(0.0, 1.0);
            thrust::host_vector<float> h_pts(opt.n_points);
            for (int i = 0; i < opt.n_points; i++) 
              h_pts[i] = 0.0001 * distribution1(generator);
            thrust::copy(h_pts.begin(), h_pts.end(), pts.begin());
            for (int i = 0; i < opt.n_points; i++) 
              h_pts[i] = 0.0001 * distribution1(generator);
            thrust::copy(h_pts.begin(), h_pts.end(), pts.begin()+nnodes+1);
        } else if (opt.initialization == BHTSNE::TSNE_INIT::RESUME) { // Preinit from vector
            // Load from vector
            if(opt.preinit_data != nullptr) {
              thrust::copy(opt.preinit_data, opt.preinit_data+(nnodes+1)*2, pts.begin());
            }
            else {
              std::cout << "E: Invalid initialization. Initialization points are null." << std::endl;
            }
        } else if (opt.initialization == BHTSNE::TSNE_INIT::VECTOR) { // Preinit from vector points only
            // Copy the pre-init data
            if(opt.preinit_data != nullptr) {
              thrust::copy(opt.preinit_data, opt.preinit_data+opt.n_points, pts.begin());
              thrust::copy(opt.preinit_data+opt.n_points+1, opt.preinit_data+opt.n_points*2 , pts.begin()+(nnodes+1));
              tsnecuda::util::GaussianNormalizeDeviceVector(dense_handle, pts, (nnodes+1), 2);
            }
            else {
              std::cout << "E: Invalid initialization. Initialization points are null." << std::endl;
            }
        } else { // Invalid initialization
            std::cout << "E: Invalid initialization type specified." << std::endl;
            exit(1);
        }

        // Initialize the learning rates and momentums
        float eta = opt.learning_rate;
        float momentum = opt.pre_exaggeration_momentum;
        float norm;
        
        // These variables currently govern the tolerance (whether it recurses on a cell)
        float theta = opt.theta;
        float epssq = opt.epssq;

        // Initialize the GPU tree memory
        InitializationKernel<<<1, 1>>>(thrust::raw_pointer_cast(errl.data()));
        GpuErrorCheck(cudaDeviceSynchronize());

    end_time = std::chrono::high_resolution_clock::now();
    times[4] = std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time).count();

    // Dump file
    float *host_ys = nullptr;
    std::ofstream dump_file;
    if (opt.get_dump_points()) {
        dump_file.open(opt.get_dump_file());
        host_ys = new float[(nnodes + 1) * 2];
        dump_file << opt.n_points << " " << 2 << std::endl;
    }

    #ifndef NO_ZMQ
    
	    bool send_zmq = opt.get_use_interactive();
	    zmq::context_t context(1);
	    zmq::socket_t publisher(context, ZMQ_REQ);
	    if (opt.get_use_interactive()) {

        // Try to connect to the socket
        if (opt.verbosity >= 1)
            std::cout << "Initializing Connection...." << std::endl;
            publisher.setsockopt(ZMQ_RCVTIMEO, opt.get_viz_timeout());
            publisher.setsockopt(ZMQ_SNDTIMEO, opt.get_viz_timeout());
        if (opt.verbosity >= 1)
            std::cout << "Waiting for connection to visualization for 10 secs...." << std::endl;
            publisher.connect(opt.get_viz_server());

            // Send the number of points we should be expecting to the server
            std::string message = std::to_string(opt.n_points);
            send_zmq = publisher.send(message.c_str(), message.length());

            // Wait for server reply
            zmq::message_t request;
            send_zmq = publisher.recv (&request);
            
            // If there's a time-out, don't bother.
            if (send_zmq) {
                if (opt.verbosity >= 1)
                    std::cout << "Visualization connected!" << std::endl;
            } else {
                std::cout << "No Visualization Terminal, continuing..." << std::endl;
                send_zmq = false;
            }
	    }
	#endif

	#ifdef NO_ZMQ
      if (opt.get_use_interactive()) 
        std::cout << "This version is not built with ZMQ for interative viz. Rebuild with WITH_ZMQ=TRUE for viz." << std::endl;
	#endif

    // Support for infinite iteration
    float attr_exaggeration = opt.early_exaggeration;

    // Random noise handling
    std::default_random_engine generator;
    std::normal_distribution<double> distribution1(0.0, 1.0);
    thrust::host_vector<float> h_pts(opt.n_points*2);
    thrust::device_vector<float> rand_noise(opt.n_points*2);
            
    for (int step = 0; step != opt.iterations; step++) {

        // Setup learning rate schedule
        if (step == opt.force_magnify_iters) {
            momentum = opt.post_exaggeration_momentum;
            attr_exaggeration = 1.0f;
        }

        // Do Force Reset
        start_time = std::chrono::high_resolution_clock::now();

            thrust::fill(attr_forces.begin(), attr_forces.end(), 0);
            thrust::fill(rep_forces.begin(), rep_forces.end(), 0);

        end_time = std::chrono::high_resolution_clock::now();
        times[5] += std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time).count();
        

        // Bounding box kernel
        start_time = std::chrono::high_resolution_clock::now();

            BoundingBoxKernel<<<blocks * FACTOR1, THREADS1>>>(nnodes, 
                                                            opt.n_points, 
                                                            thrust::raw_pointer_cast(startl.data()), 
                                                            thrust::raw_pointer_cast(childl.data()), 
                                                            thrust::raw_pointer_cast(massl.data()), 
                                                            thrust::raw_pointer_cast(pts.data()), 
                                                            thrust::raw_pointer_cast(pts.data() + nnodes + 1), 
                                                            thrust::raw_pointer_cast(maxxl.data()), 
                                                            thrust::raw_pointer_cast(maxyl.data()), 
                                                            thrust::raw_pointer_cast(minxl.data()), 
                                                            thrust::raw_pointer_cast(minyl.data()));

            GpuErrorCheck(cudaDeviceSynchronize());

        end_time = std::chrono::high_resolution_clock::now();
        times[6] += std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time).count();

        // Tree Building
        start_time = std::chrono::high_resolution_clock::now();

            ClearKernel1<<<blocks * 1, 1024>>>(nnodes, opt.n_points, thrust::raw_pointer_cast(childl.data()));
            TreeBuildingKernel<<<blocks * FACTOR2, THREADS2>>>(nnodes, opt.n_points, thrust::raw_pointer_cast(errl.data()), 
                                                                                thrust::raw_pointer_cast(childl.data()), 
                                                                                thrust::raw_pointer_cast(pts.data()), 
                                                                                thrust::raw_pointer_cast(pts.data() + nnodes + 1));
            ClearKernel2<<<blocks * 1, 1024>>>(nnodes, thrust::raw_pointer_cast(startl.data()), thrust::raw_pointer_cast(massl.data()));
            GpuErrorCheck(cudaDeviceSynchronize());

        end_time = std::chrono::high_resolution_clock::now();
        times[7] += std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time).count();

        // Tree Summarization
        start_time =  std::chrono::high_resolution_clock::now();
        
            SummarizationKernel<<<blocks * FACTOR3, THREADS3>>>(nnodes, opt.n_points, thrust::raw_pointer_cast(countl.data()), 
                                                                                        thrust::raw_pointer_cast(childl.data()), 
                                                                                        thrust::raw_pointer_cast(massl.data()),
                                                                                        thrust::raw_pointer_cast(pts.data()),
                                                                                        thrust::raw_pointer_cast(pts.data() + nnodes + 1));
            GpuErrorCheck(cudaDeviceSynchronize());

        end_time = std::chrono::high_resolution_clock::now();
        times[8] += std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time).count();

        // Force sorting
        start_time = std::chrono::high_resolution_clock::now();
        
            SortKernel<<<blocks * FACTOR4, THREADS4>>>(nnodes, opt.n_points, thrust::raw_pointer_cast(sortl.data()), 
                                                                        thrust::raw_pointer_cast(countl.data()), 
                                                                        thrust::raw_pointer_cast(startl.data()), 
                                                                        thrust::raw_pointer_cast(childl.data()));
            GpuErrorCheck(cudaDeviceSynchronize());

        end_time = std::chrono::high_resolution_clock::now();
        times[9] += std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time).count();

        // Repulsive force calculation
        start_time = std::chrono::high_resolution_clock::now();
        
            ForceCalculationKernel<<<blocks * FACTOR5, THREADS5>>>(nnodes, opt.n_points, thrust::raw_pointer_cast(errl.data()), 
                                                                        theta, epssq,
                                                                        thrust::raw_pointer_cast(sortl.data()), 
                                                                        thrust::raw_pointer_cast(childl.data()), 
                                                                        thrust::raw_pointer_cast(massl.data()), 
                                                                        thrust::raw_pointer_cast(pts.data()),
                                                                        thrust::raw_pointer_cast(pts.data() + nnodes + 1),
                                                                        thrust::raw_pointer_cast(rep_forces.data()),
                                                                        thrust::raw_pointer_cast(rep_forces.data() + nnodes + 1),
                                                                        thrust::raw_pointer_cast(norml.data()));
            GpuErrorCheck(cudaDeviceSynchronize());

        end_time = std::chrono::high_resolution_clock::now();
        times[10] += std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time).count();

        // Attractive Force Computation
        start_time = std::chrono::high_resolution_clock::now();

            // compute attractive forces
            computeAttrForce(opt.n_points, sparsePij.size(), nnodes, attr_forces_grid_size, attr_forces_block_size, sparse_handle, descr, sparsePij, pijRowPtr, pijColInd, forceProd, pts, attr_forces, ones, indices);
            GpuErrorCheck(cudaDeviceSynchronize());

        end_time = std::chrono::high_resolution_clock::now();
        times[11] += std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time).count();


        // Move the particles
        start_time = std::chrono::high_resolution_clock::now();
        
            // Compute the normalization constant
            norm = thrust::reduce(norml.begin(), norml.end(), 0.0f, thrust::plus<float>());

            // Integrate
            IntegrationKernel<<<blocks * FACTOR6, THREADS6>>>(opt.n_points, nnodes, eta, norm, momentum, attr_exaggeration,
                                                                        thrust::raw_pointer_cast(pts.data()),
                                                                        thrust::raw_pointer_cast(attr_forces.data()),
                                                                        thrust::raw_pointer_cast(rep_forces.data()),
                                                                        thrust::raw_pointer_cast(gains.data()),
                                                                        thrust::raw_pointer_cast(old_forces.data()));
            for (int i = 0; i < opt.n_points*2; i++) 
              h_pts[i] = 0.001 * distribution1(generator);
            GpuErrorCheck(cudaDeviceSynchronize());
            thrust::copy(h_pts.begin(), h_pts.end(), rand_noise.begin());

            // Compute the gradient norm
            tsnecuda::util::SquareDeviceVector(attr_forces, old_forces);
            thrust::transform(attr_forces.begin(), attr_forces.begin()+opt.n_points, 
                              attr_forces.begin()+opt.n_points, attr_forces.begin(), thrust::plus<float>());
            tsnecuda::util::SqrtDeviceVector(attr_forces, attr_forces);
            float grad_norm = thrust::reduce(attr_forces.begin(), attr_forces.begin()+opt.n_points, 0.0f, thrust::plus<float>()) / opt.n_points;

            if (opt.verbosity >= 1 && step % opt.print_interval == 0)
              std::cout << "[Step " << step << "] Average Gradient Norm: " << grad_norm << std::endl;
                                                            
            // Add some random noise to the points
            thrust::transform(pts.begin(), pts.begin()+opt.n_points, rand_noise.begin(), pts.begin(), thrust::plus<float>());
            thrust::transform(pts.begin()+nnodes+1, pts.begin()+nnodes+1+opt.n_points, rand_noise.begin()+opt.n_points, pts.begin()+nnodes+1, thrust::plus<float>());

        end_time = std::chrono::high_resolution_clock::now();
        times[12] += std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time).count();

        #ifndef NO_ZMQ
            if (send_zmq) {
            zmq::message_t message(sizeof(float)*opt.n_points*2);
            thrust::copy(pts.begin(), pts.begin()+opt.n_points, static_cast<float*>(message.data()));
            thrust::copy(pts.begin()+nnodes+1, pts.begin()+nnodes+1+opt.n_points, static_cast<float*>(message.data())+opt.n_points);
            bool res = false;
            res = publisher.send(message);
            zmq::message_t request;
            res = publisher.recv(&request);
            if (!res) {
                std::cout << "Server Disconnected, Not sending anymore for this session." << std::endl;
            }
            send_zmq = res;
            }
        #endif

        if (opt.get_dump_points() && step % opt.get_dump_interval() == 0) {
            thrust::copy(pts.begin(), pts.end(), host_ys);
            for (int i = 0; i < opt.n_points; i++) {
                dump_file << host_ys[i] << " " << host_ys[i + nnodes + 1] << std::endl;
            }
        }

        // Handle snapshoting
        if (opt.return_style == BHTSNE::RETURN_STYLE::SNAPSHOT && step % snap_interval == 0 && opt.return_data != nullptr) {
          thrust::copy(pts.begin(),
                       pts.begin()+opt.n_points, 
                       snap_num*opt.n_points*2 + opt.return_data);
          thrust::copy(pts.begin()+nnodes+1, 
                       pts.begin()+nnodes+1+opt.n_points,
                       snap_num*opt.n_points*2 + opt.return_data+opt.n_points);
          snap_num += 1;
        }

    }

    // Clean up the dump file if we are dumping points
    if (opt.get_dump_points()){
      delete[] host_ys;
      dump_file.close();
    }

    // With verbosity 2, print the timing data
    if (opt.verbosity >= 2) {
      int p1_time = times[0] + times[1] + times[2] + times[3];
      int p2_time = times[4] + times[5] + times[6] + times[7] + times[8] + times[9] + times[10] + times[11] + times[12];
      std::cout << "Timing data: " << std::endl;
      std::cout << "\t Phase 1 (" << p1_time  << "us):" << std::endl;
      std::cout << "\t\tKernel Setup: " << times[0] << "us" << std::endl;
      std::cout << "\t\tKNN Computation: " << times[1] << "us" << std::endl;
      std::cout << "\t\tPIJ Computation: " << times[2] << "us" << std::endl;
      std::cout << "\t\tPIJ Symmetrization: " << times[3] << "us" << std::endl;
      std::cout << "\t Phase 2 (" << p2_time << "us):" << std::endl;
      std::cout << "\t\tKernel Setup: " << times[4] << "us" << std::endl;
      std::cout << "\t\tForce Reset: " << times[5] << "us" << std::endl;
      std::cout << "\t\tBounding Box: " << times[6] << "us" << std::endl;
      std::cout << "\t\tTree Building: " << times[7] << "us" << std::endl;
      std::cout << "\t\tTree Summarization: " << times[8] << "us" << std::endl;
      std::cout << "\t\tSorting: " << times[9] << "us" << std::endl;
      std::cout << "\t\tRepulsive Force Calculation: " << times[10] << "us" << std::endl;
      std::cout << "\t\tAttractive Force Calculation: " << times[11] << "us" << std::endl;
      std::cout << "\t\tIntegration: " << times[12] << "us" << std::endl;
      std::cout << "Total Time: " << p1_time + p2_time << "us" << std::endl << std::endl;
    }
    // std::cout << FACTOR1 << "," << FACTOR2 << "," << FACTOR3 << "," << FACTOR4 << "," << FACTOR5 << "," << FACTOR6 <<std::endl;
    // std::cout << THREADS1 << "," << THREADS2 << "," << THREADS3 << "," << THREADS4 << "," << THREADS5 << "," << THREADS6 << std::endl;

    if (opt.verbosity >= 1) std::cout << "Fin." << std::endl;
    
    // Handle a once off return type
    if (opt.return_style == BHTSNE::RETURN_STYLE::ONCE && opt.return_data != nullptr) {
      thrust::copy(pts.begin(), pts.begin()+opt.n_points, opt.return_data);
      thrust::copy(pts.begin()+nnodes+1, pts.begin()+nnodes+1+opt.n_points, opt.return_data+opt.n_points);
    }

    // Handle snapshoting
    if (opt.return_style == BHTSNE::RETURN_STYLE::SNAPSHOT && opt.return_data != nullptr) {
      thrust::copy(pts.begin(), pts.begin()+opt.n_points, snap_num*opt.n_points*2 + opt.return_data);
      thrust::copy(pts.begin()+nnodes+1, pts.begin()+nnodes+1+opt.n_points, snap_num*opt.n_points*2 + opt.return_data+opt.n_points);
    }

    // Return some final values
    opt.trained = true;
    opt.trained_norm = norm;

    return;
}

