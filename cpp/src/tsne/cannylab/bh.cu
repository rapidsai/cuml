/*
ECL-BH v4.5: Simulation of the gravitational forces in a star cluster using
the Barnes-Hut n-body algorithm.

Copyright (c) 2010-2020 Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
   * Neither the name of Texas State University nor the names of its
     contributors may be used to endorse or promote products derived from
     this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Martin Burtscher and Sahar Azimi

URL: The latest version of this code is available at
https://userweb.cs.txstate.edu/~burtscher/research/ECL-BH/.

Publication: This work is described in detail in the following paper.
Martin Burtscher and Keshav Pingali. An Efficient CUDA Implementation of the
Tree-based Barnes Hut n-Body Algorithm. Chapter 6 in GPU Computing Gems
Emerald Edition, pp. 75-92. January 2011.
*/

#include <cuda.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// threads per block
#define THREADS1 1024 /* must be a power of 2 */
#define THREADS2 1024
#define THREADS3 768 /* shared-memory limited on some devices */
#define THREADS4 1024
#define THREADS5 1024
#define THREADS6 1024

// block count = factor * #SMs
#define FACTOR1 2
#define FACTOR2 2
#define FACTOR3 1 /* must all be resident at the same time */
#define FACTOR4 1 /* must all be resident at the same time */
#define FACTOR5 2
#define FACTOR6 2

#define WARPSIZE 32
#define MAXDEPTH 32

__device__ volatile int stepd, bottomd;
__device__ unsigned int blkcntd;
__device__ volatile float radiusd;

/******************************************************************************/
/*** initialize memory ********************************************************/
/******************************************************************************/

CUML_KERNEL void InitializationKernel()
{
  stepd   = -1;
  blkcntd = 0;
}

/******************************************************************************/
/*** compute center and radius ************************************************/
/******************************************************************************/

CUML_KERNEL __launch_bounds__(THREADS1,
                              FACTOR1) void BoundingBoxKernel(const int nnodesd,
                                                              const int nbodiesd,
                                                              int* const __restrict__ startd,
                                                              int* const __restrict__ childd,
                                                              float4* const __restrict__ posMassd,
                                                              float3* const __restrict__ maxd,
                                                              float3* const __restrict__ mind)
{
  int i, j, k, inc;
  float val;
  __shared__ volatile float sminx[THREADS1], smaxx[THREADS1], sminy[THREADS1], smaxy[THREADS1],
    sminz[THREADS1], smaxz[THREADS1];
  float3 min, max;

  // initialize with valid data (in case #bodies < #threads)
  const float4 p0 = posMassd[0];
  min.x = max.x = p0.x;
  min.y = max.y = p0.y;
  min.z = max.z = p0.z;

  // scan all bodies
  i   = threadIdx.x;
  inc = THREADS1 * gridDim.x;
  for (j = i + blockIdx.x * THREADS1; j < nbodiesd; j += inc) {
    const float4 p = posMassd[j];
    val            = p.x;
    min.x          = fminf(min.x, val);
    max.x          = fmaxf(max.x, val);
    val            = p.y;
    min.y          = fminf(min.y, val);
    max.y          = fmaxf(max.y, val);
    val            = p.z;
    min.z          = fminf(min.z, val);
    max.z          = fmaxf(max.z, val);
  }

  // reduction in shared memory
  sminx[i] = min.x;
  smaxx[i] = max.x;
  sminy[i] = min.y;
  smaxy[i] = max.y;
  sminz[i] = min.z;
  smaxz[i] = max.z;

  for (j = THREADS1 / 2; j > 0; j /= 2) {
    __syncthreads();
    if (i < j) {
      k        = i + j;
      sminx[i] = min.x = fminf(min.x, sminx[k]);
      smaxx[i] = max.x = fmaxf(max.x, smaxx[k]);
      sminy[i] = min.y = fminf(min.y, sminy[k]);
      smaxy[i] = max.y = fmaxf(max.y, smaxy[k]);
      sminz[i] = min.z = fminf(min.z, sminz[k]);
      smaxz[i] = max.z = fmaxf(max.z, smaxz[k]);
    }
  }

  // write block result to global memory
  if (i == 0) {
    k       = blockIdx.x;
    mind[k] = min;
    maxd[k] = max;
    __threadfence();

    inc = gridDim.x - 1;
    if (inc == atomicInc(&blkcntd, inc)) {
      // I'm the last block, so combine all block results
      for (j = 0; j <= inc; j++) {
        float3 minp = mind[j];
        float3 maxp = maxd[j];
        min.x       = fminf(min.x, minp.x);
        max.x       = fmaxf(max.x, maxp.x);
        min.y       = fminf(min.y, minp.y);
        max.y       = fmaxf(max.y, maxp.y);
        min.z       = fminf(min.z, minp.z);
        max.z       = fmaxf(max.z, maxp.z);
      }

      // compute radius
      val     = fmaxf(max.x - min.x, max.y - min.y);
      radiusd = fmaxf(val, max.z - min.z) * 0.5f;

      // create root node
      k       = nnodesd;
      bottomd = k;

      startd[k] = 0;
      float4 p;
      p.x         = (min.x + max.x) * 0.5f;
      p.y         = (min.y + max.y) * 0.5f;
      p.z         = (min.z + max.z) * 0.5f;
      p.w         = -1.0f;
      posMassd[k] = p;
      k *= 8;
      for (i = 0; i < 8; i++)
        childd[k + i] = -1;

      stepd++;
    }
  }
}

/******************************************************************************/
/*** build tree ***************************************************************/
/******************************************************************************/

CUML_KERNEL __launch_bounds__(1024, 1) void ClearKernel1(const int nnodesd,
                                                         const int nbodiesd,
                                                         int* const __restrict__ childd)
{
  int k, inc, top, bottom;

  top    = 8 * nnodesd;
  bottom = 8 * nbodiesd;
  inc    = blockDim.x * gridDim.x;
  k      = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;  // align to warp size
  if (k < bottom) k += inc;

  // iterate over all cells assigned to thread
  while (k < top) {
    childd[k] = -1;
    k += inc;
  }
}

CUML_KERNEL __launch_bounds__(THREADS2, FACTOR2) void TreeBuildingKernel(
  const int nnodesd,
  const int nbodiesd,
  volatile int* const __restrict__ childd,
  const float4* const __restrict__ posMassd)
{
  int i, j, depth, skip, inc;
  float x, y, z, r;
  float dx, dy, dz;
  int ch, n, cell, locked, patch;
  float radius;

  // cache root data
  radius            = radiusd * 0.5f;
  const float4 root = posMassd[nnodesd];

  skip = 1;
  inc  = blockDim.x * gridDim.x;
  i    = threadIdx.x + blockIdx.x * blockDim.x;

  // iterate over all bodies assigned to thread
  while (i < nbodiesd) {
    const float4 p = posMassd[i];
    if (skip != 0) {
      // new body, so start traversing at root
      skip  = 0;
      n     = nnodesd;
      depth = 1;
      r     = radius;
      dx = dy = dz = -r;
      j            = 0;
      // determine which child to follow
      if (root.x < p.x) {
        j  = 1;
        dx = r;
      }
      if (root.y < p.y) {
        j |= 2;
        dy = r;
      }
      if (root.z < p.z) {
        j |= 4;
        dz = r;
      }
      x = root.x + dx;
      y = root.y + dy;
      z = root.z + dz;
    }

    // follow path to leaf cell
    ch = childd[n * 8 + j];
    while (ch >= nbodiesd) {
      n = ch;
      depth++;
      r *= 0.5f;
      dx = dy = dz = -r;
      j            = 0;
      // determine which child to follow
      if (x < p.x) {
        j  = 1;
        dx = r;
      }
      if (y < p.y) {
        j |= 2;
        dy = r;
      }
      if (z < p.z) {
        j |= 4;
        dz = r;
      }
      x += dx;
      y += dy;
      z += dz;
      ch = childd[n * 8 + j];
    }

    if (ch != -2) {  // skip if child pointer is locked and try again later
      locked = n * 8 + j;
      if (ch == -1) {
        if (-1 == atomicCAS((int*)&childd[locked], -1, i)) {  // if null, just insert the new body
          i += inc;                                           // move on to next body
          skip = 1;
        }
      } else {  // there already is a body at this position
        if (ch == atomicCAS((int*)&childd[locked], ch, -2)) {  // try to lock
          patch            = -1;
          const float4 chp = posMassd[ch];
          // create new cell(s) and insert the old and new bodies
          do {
            depth++;
            if (depth > MAXDEPTH) {
              printf("ERROR: maximum depth exceeded (bodies are too close together)\n");
              asm("trap;");
            }

            cell = atomicSub((int*)&bottomd, 1) - 1;
            if (cell <= nbodiesd) {
              printf("ERROR: out of cell memory\n");
              asm("trap;");
            }

            if (patch != -1) { childd[n * 8 + j] = cell; }
            patch = max(patch, cell);

            j = 0;
            if (x < chp.x) j = 1;
            if (y < chp.y) j |= 2;
            if (z < chp.z) j |= 4;
            childd[cell * 8 + j] = ch;

            n = cell;
            r *= 0.5f;
            dx = dy = dz = -r;
            j            = 0;
            if (x < p.x) {
              j  = 1;
              dx = r;
            }
            if (y < p.y) {
              j |= 2;
              dy = r;
            }
            if (z < p.z) {
              j |= 4;
              dz = r;
            }
            x += dx;
            y += dy;
            z += dz;

            ch = childd[n * 8 + j];
            // repeat until the two bodies are different children
          } while (ch >= 0);
          childd[n * 8 + j] = i;

          i += inc;  // move on to next body
          skip = 2;
        }
      }
    }
    __syncthreads();  // optional barrier for performance
    __threadfence();

    if (skip == 2) { childd[locked] = patch; }
  }
}

CUML_KERNEL __launch_bounds__(1024, 1) void ClearKernel2(const int nnodesd,
                                                         int* const __restrict__ startd,
                                                         float4* const __restrict__ posMassd)
{
  int k, inc, bottom;

  bottom = bottomd;
  inc    = blockDim.x * gridDim.x;
  k      = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;  // align to warp size
  if (k < bottom) k += inc;

  // iterate over all cells assigned to thread
  while (k < nnodesd) {
    posMassd[k].w = -1.0f;
    startd[k]     = -1;
    k += inc;
  }
}

/******************************************************************************/
/*** compute center of mass ***************************************************/
/******************************************************************************/

CUML_KERNEL __launch_bounds__(THREADS3, FACTOR3) void SummarizationKernel(
  const int nnodesd,
  const int nbodiesd,
  volatile int* const __restrict__ countd,
  const int* const __restrict__ childd,
  volatile float4* const __restrict__ posMassd)
{
  int i, j, k, ch, inc, cnt, bottom;
  float m, cm, px, py, pz;
  __shared__ int child[THREADS3 * 8];
  __shared__ float mass[THREADS3 * 8];

  bottom = bottomd;
  inc    = blockDim.x * gridDim.x;
  k      = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;  // align to warp size
  if (k < bottom) k += inc;

  int restart = k;
  for (j = 0; j < 3; j++) {  // wait-free pre-passes
    // iterate over all cells assigned to thread
    while (k <= nnodesd) {
      if (posMassd[k].w < 0.0f) {
        for (i = 0; i < 8; i++) {
          ch                                = childd[k * 8 + i];
          child[i * THREADS3 + threadIdx.x] = ch;  // cache children
          if ((ch >= nbodiesd) && ((mass[i * THREADS3 + threadIdx.x] = posMassd[ch].w) < 0.0f)) {
            break;
          }
        }
        if (i == 8) {
          // all children are ready
          cm  = 0.0f;
          px  = 0.0f;
          py  = 0.0f;
          pz  = 0.0f;
          cnt = 0;
          for (i = 0; i < 8; i++) {
            ch = child[i * THREADS3 + threadIdx.x];
            if (ch >= 0) {
              // four reads due to missing copy constructor for "volatile float4"
              const float chx = posMassd[ch].x;
              const float chy = posMassd[ch].y;
              const float chz = posMassd[ch].z;
              const float chw = posMassd[ch].w;
              if (ch >= nbodiesd) {  // count bodies (needed later)
                m = mass[i * THREADS3 + threadIdx.x];
                cnt += countd[ch];
              } else {
                m = chw;
                cnt++;
              }
              // add child's contribution
              cm += m;
              px += chx * m;
              py += chy * m;
              pz += chz * m;
            }
          }
          countd[k] = cnt;
          m         = 1.0f / cm;
          // four writes due to missing copy constructor for "volatile float4"
          posMassd[k].x = px * m;
          posMassd[k].y = py * m;
          posMassd[k].z = pz * m;
          __threadfence();
          posMassd[k].w = cm;
        }
      }
      k += inc;  // move on to next cell
    }
    k = restart;
  }

  j = 0;
  // iterate over all cells assigned to thread
  while (k <= nnodesd) {
    if (posMassd[k].w >= 0.0f) {
      k += inc;
    } else {
      if (j == 0) {
        j = 8;
        for (i = 0; i < 8; i++) {
          ch                                = childd[k * 8 + i];
          child[i * THREADS3 + threadIdx.x] = ch;  // cache children
          if ((ch < nbodiesd) || ((mass[i * THREADS3 + threadIdx.x] = posMassd[ch].w) >= 0.0f)) {
            j--;
          }
        }
      } else {
        j = 8;
        for (i = 0; i < 8; i++) {
          ch = child[i * THREADS3 + threadIdx.x];
          if ((ch < nbodiesd) || (mass[i * THREADS3 + threadIdx.x] >= 0.0f) ||
              ((mass[i * THREADS3 + threadIdx.x] = posMassd[ch].w) >= 0.0f)) {
            j--;
          }
        }
      }

      if (j == 0) {
        // all children are ready
        cm  = 0.0f;
        px  = 0.0f;
        py  = 0.0f;
        pz  = 0.0f;
        cnt = 0;
        for (i = 0; i < 8; i++) {
          ch = child[i * THREADS3 + threadIdx.x];
          if (ch >= 0) {
            // four reads due to missing copy constructor for "volatile float4"
            const float chx = posMassd[ch].x;
            const float chy = posMassd[ch].y;
            const float chz = posMassd[ch].z;
            const float chw = posMassd[ch].w;
            if (ch >= nbodiesd) {  // count bodies (needed later)
              m = mass[i * THREADS3 + threadIdx.x];
              cnt += countd[ch];
            } else {
              m = chw;
              cnt++;
            }
            // add child's contribution
            cm += m;
            px += chx * m;
            py += chy * m;
            pz += chz * m;
          }
        }
        countd[k] = cnt;
        m         = 1.0f / cm;
        // four writes due to missing copy constructor for "volatile float4"
        posMassd[k].x = px * m;
        posMassd[k].y = py * m;
        posMassd[k].z = pz * m;
        __threadfence();
        posMassd[k].w = cm;
        k += inc;
      }
    }
  }
}

/******************************************************************************/
/*** sort bodies **************************************************************/
/******************************************************************************/

CUML_KERNEL __launch_bounds__(THREADS4,
                              FACTOR4) void SortKernel(const int nnodesd,
                                                       const int nbodiesd,
                                                       int* const __restrict__ sortd,
                                                       const int* const __restrict__ countd,
                                                       volatile int* const __restrict__ startd,
                                                       int* const __restrict__ childd)
{
  int i, j, k, ch, dec, start, bottom;

  bottom = bottomd;
  dec    = blockDim.x * gridDim.x;
  k      = nnodesd + 1 - dec + threadIdx.x + blockIdx.x * blockDim.x;

  // iterate over all cells assigned to thread
  while (k >= bottom) {
    start = startd[k];
    if (start >= 0) {
      j = 0;
      for (i = 0; i < 8; i++) {
        ch = childd[k * 8 + i];
        if (ch >= 0) {
          if (i != j) {
            // move children to front (needed later for speed)
            childd[k * 8 + i] = -1;
            childd[k * 8 + j] = ch;
          }
          j++;
          if (ch >= nbodiesd) {
            // child is a cell
            startd[ch] = start;   // set start ID of child
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
    __syncthreads();  // optional barrier for performance
  }
}

/******************************************************************************/
/*** compute force ************************************************************/
/******************************************************************************/

CUML_KERNEL __launch_bounds__(THREADS5, FACTOR5) void ForceCalculationKernel(
  const int nnodesd,
  const int nbodiesd,
  const float dthfd,
  const float itolsqd,
  const float epssqd,
  const int* const __restrict__ sortd,
  const int* const __restrict__ childd,
  const float4* const __restrict__ posMassd,
  float2* const __restrict__ veld,
  float4* const __restrict__ accVeld)
{
  int i, j, k, n, depth, base, sbase, diff, pd, nd;
  float ax, ay, az, dx, dy, dz, tmp;
  __shared__ volatile int pos[MAXDEPTH * THREADS5 / WARPSIZE], node[MAXDEPTH * THREADS5 / WARPSIZE];
  __shared__ float dq[MAXDEPTH * THREADS5 / WARPSIZE];

  if (0 == threadIdx.x) {
    tmp = radiusd * 2;
    // precompute values that depend only on tree level
    dq[0] = tmp * tmp * itolsqd;
    for (i = 1; i < MAXDEPTH; i++) {
      dq[i] = dq[i - 1] * 0.25f;
      dq[i - 1] += epssqd;
    }
    dq[i - 1] += epssqd;
  }
  __syncthreads();

  // figure out first thread in each warp (lane 0)
  base  = threadIdx.x / WARPSIZE;
  sbase = base * WARPSIZE;
  j     = base * MAXDEPTH;

  diff = threadIdx.x - sbase;
  // make multiple copies to avoid index calculations later
  if (diff < MAXDEPTH) { dq[diff + j] = dq[diff]; }
  __syncthreads();

  // iterate over all bodies assigned to thread
  for (k = threadIdx.x + blockIdx.x * blockDim.x; k < nbodiesd; k += blockDim.x * gridDim.x) {
    i = sortd[k];  // get permuted/sorted index
    // cache position info
    const float4 pi = posMassd[i];

    ax = 0.0f;
    ay = 0.0f;
    az = 0.0f;

    // initialize iteration stack, i.e., push root node onto stack
    depth = j;
    if (sbase == threadIdx.x) {
      pos[j]  = 0;
      node[j] = nnodesd * 8;
    }

    do {
      // stack is not empty
      pd = pos[depth];
      nd = node[depth];
      while (pd < 8) {
        // node on top of stack has more children to process
        n = childd[nd + pd];  // load child pointer
        pd++;

        if (n >= 0) {
          const float4 pn = posMassd[n];
          dx              = pn.x - pi.x;
          dy              = pn.y - pi.y;
          dz              = pn.z - pi.z;
          tmp =
            dx * dx + (dy * dy + (dz * dz + epssqd));  // compute distance squared (plus softening)
          if ((n < nbodiesd) ||
              __all_sync(0xffffffff, tmp >= dq[depth])) {  // check if all threads agree that cell
                                                           // is far enough away (or is a body)
            tmp = rsqrtf(tmp);                             // compute distance
            tmp = pn.w * tmp * tmp * tmp;
            ax += dx * tmp;
            ay += dy * tmp;
            az += dz * tmp;
          } else {
            // push cell onto stack
            if (sbase == threadIdx.x) {
              pos[depth]  = pd;
              node[depth] = nd;
            }
            depth++;
            pd = 0;
            nd = n * 8;
          }
        } else {
          pd = 8;  // early out because all remaining children are also zero
        }
      }
      depth--;  // done with this level
    } while (depth >= j);

    float4 acc = accVeld[i];
    if (stepd > 0) {
      // update velocity
      float2 v = veld[i];
      v.x += (ax - acc.x) * dthfd;
      v.y += (ay - acc.y) * dthfd;
      acc.w += (az - acc.z) * dthfd;
      veld[i] = v;
    }

    // save computed acceleration
    acc.x      = ax;
    acc.y      = ay;
    acc.z      = az;
    accVeld[i] = acc;
  }
}

/******************************************************************************/
/*** advance bodies ***********************************************************/
/******************************************************************************/

CUML_KERNEL __launch_bounds__(THREADS6,
                              FACTOR6) void IntegrationKernel(const int nbodiesd,
                                                              const float dtimed,
                                                              const float dthfd,
                                                              float4* const __restrict__ posMass,
                                                              float2* const __restrict__ veld,
                                                              float4* const __restrict__ accVeld)
{
  int i, inc;
  float dvelx, dvely, dvelz;
  float velhx, velhy, velhz;

  // iterate over all bodies assigned to thread
  inc = blockDim.x * gridDim.x;
  for (i = threadIdx.x + blockIdx.x * blockDim.x; i < nbodiesd; i += inc) {
    // integrate
    float4 acc = accVeld[i];
    dvelx      = acc.x * dthfd;
    dvely      = acc.y * dthfd;
    dvelz      = acc.z * dthfd;

    float2 v = veld[i];
    velhx    = v.x + dvelx;
    velhy    = v.y + dvely;
    velhz    = acc.w + dvelz;

    float4 p = posMass[i];
    p.x += velhx * dtimed;
    p.y += velhy * dtimed;
    p.z += velhz * dtimed;
    posMass[i] = p;

    v.x        = velhx + dvelx;
    v.y        = velhy + dvely;
    acc.w      = velhz + dvelz;
    veld[i]    = v;
    accVeld[i] = acc;
  }
}

/******************************************************************************/

static void CudaTest(const char* const msg)
{
  cudaError_t e;

  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "%s: %d\n", msg, e);
    fprintf(stderr, "%s\n", cudaGetErrorString(e));
    exit(-1);
  }
}

/******************************************************************************/

// random number generator (based on SPLASH-2 code at
// https://github.com/staceyson/splash2/blob/master/codes/apps/barnes/util.C)

static int randx = 7;

static double drnd()
{
  const int lastrand = randx;
  randx              = (1103515245 * randx + 12345) & 0x7FFFFFFF;
  return (double)lastrand / 2147483648.0;
}

/******************************************************************************/

int main(int argc, char* argv[])
{
  int i, run, blocks;
  int nnodes, nbodies, step, timesteps;
  double runtime;
  float dtime, dthf, epssq, itolsq;
  float time, timing[7];
  cudaEvent_t start, stop;

  float4* accVel;
  float2* vel;
  int *sortl, *childl, *countl, *startl;
  float4* accVell;
  float2* vell;
  float3 *maxl, *minl;
  float4* posMassl;
  float4* posMass;
  double rsc, vsc, r, v, x, y, z, sq, scale;

  // perform some checks

  printf("ECL-BH v4.5\n");
  printf("Copyright (c) 2010-2020 Texas State University\n");
  fflush(stdout);

  if (argc != 4) {
    fprintf(stderr, "\n");
    fprintf(stderr, "arguments: number_of_bodies number_of_timesteps device\n");
    exit(-1);
  }

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    fprintf(stderr, "There is no device supporting CUDA\n");
    exit(-1);
  }

  const int dev = atoi(argv[3]);
  if ((dev < 0) || (deviceCount <= dev)) {
    fprintf(stderr, "There is no device %d\n", dev);
    exit(-1);
  }
  cudaSetDevice(dev);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
    fprintf(stderr, "There is no CUDA capable device\n");
    exit(-1);
  }
  if (deviceProp.major < 3) {
    fprintf(stderr, "Need at least compute capability 3.0\n");
    exit(-1);
  }
  if (deviceProp.warpSize != WARPSIZE) {
    fprintf(stderr, "Warp size must be %d\n", deviceProp.warpSize);
    exit(-1);
  }

  blocks         = deviceProp.multiProcessorCount;
  const int mTSM = deviceProp.maxThreadsPerMultiProcessor;
  printf("gpu: %s with %d SMs and %d mTpSM (%.1f MHz and %.1f MHz)\n",
         deviceProp.name,
         blocks,
         mTSM,
         deviceProp.clockRate * 0.001,
         deviceProp.memoryClockRate * 0.001);

  if ((WARPSIZE <= 0) || (WARPSIZE & (WARPSIZE - 1) != 0)) {
    fprintf(stderr, "Warp size must be greater than zero and a power of two\n");
    exit(-1);
  }
  if (MAXDEPTH > WARPSIZE) {
    fprintf(stderr, "MAXDEPTH must be less than or equal to WARPSIZE\n");
    exit(-1);
  }
  if ((THREADS1 <= 0) || (THREADS1 & (THREADS1 - 1) != 0)) {
    fprintf(stderr, "THREADS1 must be greater than zero and a power of two\n");
    exit(-1);
  }

  // set L1/shared memory configuration
  cudaFuncSetCacheConfig(BoundingBoxKernel, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(TreeBuildingKernel, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(ClearKernel1, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(ClearKernel2, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(SummarizationKernel, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(SortKernel, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(ForceCalculationKernel, cudaFuncCachePreferEqual);
  cudaFuncSetCacheConfig(IntegrationKernel, cudaFuncCachePreferL1);

  cudaGetLastError();              // reset error value
  for (run = 0; run < 1; run++) {  // in case multiple runs are desired for timing
    for (i = 0; i < 7; i++)
      timing[i] = 0.0f;

    nbodies = atoi(argv[1]);
    if (nbodies < 1) {
      fprintf(stderr, "nbodies is too small: %d\n", nbodies);
      exit(-1);
    }
    if (nbodies > (1 << 30)) {
      fprintf(stderr, "nbodies is too large: %d\n", nbodies);
      exit(-1);
    }
    nnodes = nbodies * 2;
    if (nnodes < 1024 * blocks) nnodes = 1024 * blocks;
    while ((nnodes & (WARPSIZE - 1)) != 0)
      nnodes++;
    nnodes--;

    timesteps = atoi(argv[2]);
    dtime     = 0.025;
    dthf      = dtime * 0.5f;
    epssq     = 0.05 * 0.05;
    itolsq    = 1.0f / (0.5 * 0.5);

    // allocate memory

    if (run == 0) {
      printf("configuration: %d bodies, %d time steps\n", nbodies, timesteps);

      accVel = (float4*)malloc(sizeof(float4) * nbodies);
      if (accVel == NULL) {
        fprintf(stderr, "cannot allocate accVel\n");
        exit(-1);
      }
      vel = (float2*)malloc(sizeof(float2) * nbodies);
      if (vel == NULL) {
        fprintf(stderr, "cannot allocate vel\n");
        exit(-1);
      }
      posMass = (float4*)malloc(sizeof(float4) * nbodies);
      if (posMass == NULL) {
        fprintf(stderr, "cannot allocate posMass\n");
        exit(-1);
      }

      if (cudaSuccess != cudaMalloc((void**)&childl, sizeof(int) * (nnodes + 1) * 8))
        fprintf(stderr, "could not allocate childd\n");
      CudaTest("couldn't allocate childd");
      if (cudaSuccess != cudaMalloc((void**)&vell, sizeof(float2) * (nnodes + 1)))
        fprintf(stderr, "could not allocate veld\n");
      CudaTest("couldn't allocate veld");
      if (cudaSuccess != cudaMalloc((void**)&accVell, sizeof(float4) * (nnodes + 1)))
        fprintf(stderr, "could not allocate accVeld\n");
      CudaTest("couldn't allocate accVeld");
      if (cudaSuccess != cudaMalloc((void**)&countl, sizeof(int) * (nnodes + 1)))
        fprintf(stderr, "could not allocate countd\n");
      CudaTest("couldn't allocate countd");
      if (cudaSuccess != cudaMalloc((void**)&startl, sizeof(int) * (nnodes + 1)))
        fprintf(stderr, "could not allocate startd\n");
      CudaTest("couldn't allocate startd");
      if (cudaSuccess != cudaMalloc((void**)&sortl, sizeof(int) * (nnodes + 1)))
        fprintf(stderr, "could not allocate sortd\n");
      CudaTest("couldn't allocate sortd");

      if (cudaSuccess != cudaMalloc((void**)&posMassl, sizeof(float4) * (nnodes + 1)))
        fprintf(stderr, "could not allocate posMassd\n");
      CudaTest("couldn't allocate posMassd");

      if (cudaSuccess != cudaMalloc((void**)&maxl, sizeof(float3) * blocks * FACTOR1))
        fprintf(stderr, "could not allocate maxd\n");
      CudaTest("couldn't allocate maxd");
      if (cudaSuccess != cudaMalloc((void**)&minl, sizeof(float3) * blocks * FACTOR1))
        fprintf(stderr, "could not allocate mind\n");
      CudaTest("couldn't allocate mind");
    }

    // generate input (based on SPLASH-2 code at
    // https://github.com/staceyson/splash2/blob/master/codes/apps/barnes/code.C)

    rsc = (3 * 3.1415926535897932384626433832795) / 16;
    vsc = sqrt(1.0 / rsc);
    for (i = 0; i < nbodies; i++) {
      float4 p;
      p.w = 1.0 / nbodies;
      r   = 1.0 / sqrt(pow(drnd() * 0.999, -2.0 / 3.0) - 1);
      do {
        x  = drnd() * 2.0 - 1.0;
        y  = drnd() * 2.0 - 1.0;
        z  = drnd() * 2.0 - 1.0;
        sq = x * x + y * y + z * z;
      } while (sq > 1.0);
      scale      = rsc * r / sqrt(sq);
      p.x        = x * scale;
      p.y        = y * scale;
      p.z        = z * scale;
      posMass[i] = p;

      do {
        x = drnd();
        y = drnd() * 0.1;
      } while (y > x * x * pow(1 - x * x, 3.5));
      v = x * sqrt(2.0 / sqrt(1 + r * r));
      do {
        x  = drnd() * 2.0 - 1.0;
        y  = drnd() * 2.0 - 1.0;
        z  = drnd() * 2.0 - 1.0;
        sq = x * x + y * y + z * z;
      } while (sq > 1.0);
      scale = vsc * v / sqrt(sq);
      float2 v;
      v.x         = x * scale;
      v.y         = y * scale;
      accVel[i].w = z * scale;
      vel[i]      = v;
    }

    if (cudaSuccess !=
        cudaMemcpy(accVell, accVel, sizeof(float4) * nbodies, cudaMemcpyHostToDevice))
      fprintf(stderr, "copying of vel to device failed\n");
    CudaTest("vel copy to device failed");
    if (cudaSuccess != cudaMemcpy(vell, vel, sizeof(float2) * nbodies, cudaMemcpyHostToDevice))
      fprintf(stderr, "copying of vel to device failed\n");
    CudaTest("vel copy to device failed");
    if (cudaSuccess !=
        cudaMemcpy(posMassl, posMass, sizeof(float4) * nbodies, cudaMemcpyHostToDevice))
      fprintf(stderr, "copying of posMass to device failed\n");
    CudaTest("posMass copy to device failed");

    // run timesteps (launch GPU kernels)

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    struct timeval starttime, endtime;
    gettimeofday(&starttime, NULL);

    cudaEventRecord(start, 0);
    InitializationKernel<<<1, 1>>>();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    timing[0] += time;
    // CudaTest("kernel 0 launch failed");

    for (step = 0; step < timesteps; step++) {
      cudaEventRecord(start, 0);
      BoundingBoxKernel<<<blocks * FACTOR1, THREADS1>>>(
        nnodes, nbodies, startl, childl, posMassl, maxl, minl);
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&time, start, stop);
      timing[1] += time;
      // CudaTest("kernel 1 launch failed");

      cudaEventRecord(start, 0);
      ClearKernel1<<<blocks * 1, 1024>>>(nnodes, nbodies, childl);
      TreeBuildingKernel<<<blocks * FACTOR2, THREADS2>>>(nnodes, nbodies, childl, posMassl);
      ClearKernel2<<<blocks * 1, 1024>>>(nnodes, startl, posMassl);
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&time, start, stop);
      timing[2] += time;
      // CudaTest("kernel 2 launch failed");

      cudaEventRecord(start, 0);
      SummarizationKernel<<<blocks * FACTOR3, THREADS3>>>(
        nnodes, nbodies, countl, childl, posMassl);
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&time, start, stop);
      timing[3] += time;
      // CudaTest("kernel 3 launch failed");

      cudaEventRecord(start, 0);
      SortKernel<<<blocks * FACTOR4, THREADS4>>>(nnodes, nbodies, sortl, countl, startl, childl);
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&time, start, stop);
      timing[4] += time;
      // CudaTest("kernel 4 launch failed");

      cudaEventRecord(start, 0);
      ForceCalculationKernel<<<blocks * FACTOR5, THREADS5>>>(
        nnodes, nbodies, dthf, itolsq, epssq, sortl, childl, posMassl, vell, accVell);
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&time, start, stop);
      timing[5] += time;
      // CudaTest("kernel 5 launch failed");

      cudaEventRecord(start, 0);
      IntegrationKernel<<<blocks * FACTOR6, THREADS6>>>(
        nbodies, dtime, dthf, posMassl, vell, accVell);
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&time, start, stop);
      timing[6] += time;
      // CudaTest("kernel 6 launch failed");
    }
    CudaTest("kernel launch failed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    gettimeofday(&endtime, NULL);
    runtime = (endtime.tv_sec + endtime.tv_usec / 1000000.0 - starttime.tv_sec -
               starttime.tv_usec / 1000000.0);

    printf("runtime: %.4lf s  (", runtime);
    time = 0;
    for (i = 1; i < 7; i++) {
      printf(" %.1f ", timing[i]);
      time += timing[i];
    }
    printf(") = %.1f ms\n", time);
  }

  // transfer final result back to CPU
  if (cudaSuccess != cudaMemcpy(accVel, accVell, sizeof(float4) * nbodies, cudaMemcpyDeviceToHost))
    fprintf(stderr, "copying of accVel from device failed\n");
  CudaTest("vel copy from device failed");
  if (cudaSuccess != cudaMemcpy(vel, vell, sizeof(float2) * nbodies, cudaMemcpyDeviceToHost))
    fprintf(stderr, "copying of vel from device failed\n");
  CudaTest("vel copy from device failed");
  if (cudaSuccess !=
      cudaMemcpy(posMass, posMassl, sizeof(float4) * nbodies, cudaMemcpyDeviceToHost))
    fprintf(stderr, "copying of posMass from device failed\n");
  CudaTest("posMass copy from device failed");

  // print output
  i = 0;
  //  for (i = 0; i < nbodies; i++) {
  printf("%.2e %.2e %.2e\n", posMass[i].x, posMass[i].y, posMass[i].z);
  //  }

  free(vel);
  free(accVel);
  free(posMass);

  cudaFree(childl);
  cudaFree(vell);
  cudaFree(accVell);
  cudaFree(countl);
  cudaFree(startl);
  cudaFree(sortl);
  cudaFree(posMassl);
  cudaFree(maxl);
  cudaFree(minl);

  return 0;
}
