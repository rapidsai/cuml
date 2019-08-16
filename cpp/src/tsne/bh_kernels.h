/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#define restrict __restrict__

#define THREADS1 512
#define THREADS2 512
#define THREADS3 768
#define THREADS4 128
#define THREADS5 1024
#define THREADS6 1024
#define THREADS7 1024

#define FACTOR1 3
#define FACTOR2 3
#define FACTOR3 1
#define FACTOR4 4
#define FACTOR5 2
#define FACTOR6 2
#define FACTOR7 1

#include <float.h>
#include <math.h>
#include "utils.h"

namespace ML {
namespace TSNE {

/**
 * Intializes the states of objects. This speeds the overall kernel up.
 */
__global__ void
InitializationKernel( /*int *restrict errd, */
                     unsigned *restrict limiter,
                     int *restrict maxdepthd,
                     float *restrict radiusd)
{
  // errd[0] = 0;
  maxdepthd[0] = 1;
  limiter[0] = 0;
  radiusd[0] = 0.0f;
}

/**
 * Reset normalization back to 0.
 */
__global__ void
Reset_Normalization(float *restrict Z_norm,
                    float *restrict radiusd_squared,
                    int *restrict bottomd, const int NNODES,
                    const float *restrict radiusd)
{
  Z_norm[0] = 0.0f;
  radiusd_squared[0] = radiusd[0] * radiusd[0];
  // create root node
  bottomd[0] = NNODES;
}

/**
 * Find 1/Z
 */
__global__ void
Find_Normalization(float *restrict Z_norm,
                   const float N)
{
  Z_norm[0] = 1.0f / (Z_norm[0] - N);
}

/**
 * Figures the bounding boxes for every point in the embedding.
 */
__global__ __launch_bounds__(THREADS1, FACTOR1) void
BoundingBoxKernel(int *restrict startd,
                  int *restrict childd,
                  float *restrict massd,
                  float *restrict posxd,
                  float *restrict posyd,
                  float *restrict maxxd,
                  float *restrict maxyd,
                  float *restrict minxd,
                  float *restrict minyd,
                  const int FOUR_NNODES,
                  const int NNODES,
                  const int N,
                  unsigned *restrict limiter,
                  float *restrict radiusd)
{
  float val, minx, maxx, miny, maxy;
  __shared__ float sminx[THREADS1], smaxx[THREADS1], sminy[THREADS1], smaxy[THREADS1];

  // initialize with valid data (in case #bodies < #threads)
  minx = maxx = posxd[0];
  miny = maxy = posyd[0];

  // scan all bodies
  const int i = threadIdx.x;
  const int inc = THREADS1 * gridDim.x;
  for (int j = i + blockIdx.x * THREADS1; j < N; j += inc)
  {
    val = posxd[j];
    if (val < minx)      minx = val;
    else if (val > maxx) maxx = val;

    val = posyd[j];
    if (val < miny)      miny = val;
    else if (val > maxy) maxy = val;
  }

  // reduction in shared memory
  sminx[i] = minx;
  smaxx[i] = maxx;
  sminy[i] = miny;
  smaxy[i] = maxy;

  for (int j = THREADS1 / 2; j > i; j /= 2)
  {
    __syncthreads();
    const int k = i + j;
    sminx[i] = minx = fminf(minx, sminx[k]);
    smaxx[i] = maxx = fmaxf(maxx, smaxx[k]);
    sminy[i] = miny = fminf(miny, sminy[k]);
    smaxy[i] = maxy = fmaxf(maxy, smaxy[k]);
  }
  

  if (i == 0)
  {
    // write block result to global memory
    const int k = blockIdx.x;
    minxd[k] = minx;
    maxxd[k] = maxx;
    minyd[k] = miny;
    maxyd[k] = maxy;
    __threadfence();

    const int inc = gridDim.x - 1;
    if (inc != atomicInc(limiter, inc)) return;

    // I'm the last block, so combine all block results
    for (int j = 0; j <= inc; j++)
    {
      minx = fminf(minx, minxd[j]);
      maxx = fmaxf(maxx, maxxd[j]);
      miny = fminf(miny, minyd[j]);
      maxy = fmaxf(maxy, maxyd[j]);
    }

    // compute 'radius'
    atomicExch(radiusd, fmaxf(maxx - minx, maxy - miny) * 0.5f + 1e-5f);

    massd[NNODES] = -1.0f;
    startd[NNODES] = 0;
    posxd[NNODES] = (minx + maxx) * 0.5f;
    posyd[NNODES] = (miny + maxy) * 0.5f;

    #pragma unroll
    for (int a = 0; a < 4; a++)
      childd[FOUR_NNODES + a] = -1;
  }
}


/**
 * Clear some of the state vectors up.
 */
__global__ __launch_bounds__(1024, 1) void
ClearKernel1(int *restrict childd,
             const int FOUR_NNODES,
             const int FOUR_N)
{
  const int inc = blockDim.x * gridDim.x;
  int k = (FOUR_N & -32) + threadIdx.x + blockIdx.x * blockDim.x; 
  if (k < FOUR_N) k += inc;

  // iterate over all cells assigned to thread
  #pragma unroll
  for (; k < FOUR_NNODES; k += inc)
    childd[k] = -1;
}


/**
 * Build the actual KD Tree.
 */
__global__ __launch_bounds__(THREADS2, FACTOR2) void
TreeBuildingKernel( /* int *restrict errd, */
                   int *restrict childd,
                   const float *restrict posxd,
                   const float *restrict posyd,
                   const int NNODES,
                   const int N,
                   int *restrict maxdepthd,
                   int *restrict bottomd,
                   const float *restrict radiusd)
{
  int j, depth;
  float x, y, r;
  float px, py;
  int ch, n, locked, patch;

  // cache root data
  const float radius = radiusd[0];
  const float rootx = posxd[NNODES];
  const float rooty = posyd[NNODES];

  int localmaxdepth = 1;
  int skip = 1;
  const int inc = blockDim.x * gridDim.x;
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  // iterate over all bodies assigned to thread
  while (i < N)
  {
    if (skip != 0)
    {
      // new body, so start traversing at root
      skip = 0;
      n = NNODES;
      depth = 1;
      r = radius * 0.5f; 

      x = rootx + ( (rootx < (px = posxd[i])) ?
                    (j = 1, r) : (j = 0, -r) );

      y = rooty + ( (rooty < (py = posyd[i])) ?
                    (j |= 2, r) : (-r) );
    }

    // follow path to leaf cell
    while ((ch = childd[n * 4 + j]) >= N)
    {
      n = ch;
      depth++;
      r *= 0.5f;

      // determine which child to follow
      x += ( (x < px) ?
             (j = 1, r) : (j = 0, -r) );

      y += ( (y < py) ?
             (j |= 2, r) : (-r) );
    }


    if (ch != -2)
    {
      // skip if child pointer is locked and try again later
      locked = n * 4 + j;

      if (ch == -1)
      {
        if (atomicCAS(&childd[locked], -1, i) == -1)
        {
          if (depth > localmaxdepth)
            localmaxdepth = depth;

          i += inc;  // move on to next body
          skip = 1;
        }
      }
      else
      {
        if (ch == atomicCAS(&childd[locked], ch, -2))
        { 
          // try to lock
          patch = -1;

          while (ch >= 0)
          {
            depth++;

            const int cell = atomicSub(bottomd, 1) - 1;
            if (cell <= N) {
              // atomicExch(errd, 1);
              atomicExch(bottomd, NNODES);
            }

            if (patch != -1)
              childd[n * 4 + j] = cell;

            if (cell > patch)
              patch = cell;

            j = (x < posxd[ch]) ? 1 : 0;
            if (y < posyd[ch])
              j |= 2;

            childd[cell * 4 + j] = ch;
            n = cell;
            r *= 0.5f;

            x += (
                (x < px) ?
                (j = 1, r) : (j = 0, -r)
              );

            y += (
                (y < py) ?
                (j |= 2, r) : (-r)
              );

            ch = childd[n * 4 + j];
            if (r <= 1e-10) break;
          }

          childd[n * 4 + j] = i;

          if (depth > localmaxdepth)
            localmaxdepth = depth;

          i += inc;  // move on to next body
          skip = 2;
        }
      }
    }
    __threadfence();

    if (skip == 2)
      childd[locked] = patch;
  }

  // record maximum tree depth
  // if (localmaxdepth >= THREADS5)
  //   localmaxdepth = THREADS5 - 1;
  if (localmaxdepth > 32)
    localmaxdepth = 32;

  atomicMax(maxdepthd, localmaxdepth);
}

/**
 * Clean more state vectors.
 */
__global__ __launch_bounds__(1024, 1) void
ClearKernel2(int *restrict startd,
             float *restrict massd,
             const int NNODES,
             const int *restrict bottomd)
{
  const int bottom = bottomd[0];
  const int inc = blockDim.x * gridDim.x;
  int k = (bottom & -32) + threadIdx.x + blockIdx.x * blockDim.x;
  if (k < bottom) k += inc;

  // iterate over all cells assigned to thread
  #pragma unroll
  for (; k < NNODES; k+= inc) {
    massd[k] = -1.0f;
    startd[k] = -1;
  }
}

/**
 * Summarize the KD Tree via cell gathering
 */
__global__ __launch_bounds__(THREADS3, FACTOR3) void
SummarizationKernel(int *restrict countd,
                    const int *restrict childd,
                    volatile float *restrict massd,
                    float *restrict posxd,
                    float *restrict posyd,
                    const int NNODES,
                    const int N,
                    const int *restrict bottomd) 
{
  bool flag = 0;
  float cm, px, py;
  __shared__ int child[THREADS3 * 4];
  __shared__ float mass[THREADS3 * 4];

  const int bottom = bottomd[0];
  const int inc = blockDim.x * gridDim.x;
  int k = (bottom & -32) + threadIdx.x + blockIdx.x * blockDim.x;
  if (k < bottom) k += inc;

  const int restart = k;

  for (int j = 0; j < 5; j++) // wait-free pre-passes
  {
    // iterate over all cells assigned to thread
    while (k <= NNODES)
    {
      if (massd[k] < 0.0f)
      {
        for (int i = 0; i < 4; i++)
        {
          const int ch = childd[k * 4 + i];
          child[i * THREADS3 + threadIdx.x] = ch;

          if ((ch >= N) and ((mass[i * THREADS3 + threadIdx.x] = massd[ch]) < 0))
            goto CONTINUE_LOOP;
        }

        // all children are ready
        cm = 0.0f;
        px = 0.0f;
        py = 0.0f;
        int cnt = 0;

        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
          const int ch = child[i * THREADS3 + threadIdx.x];
          if (ch >= 0)
          {
            const float m =
              (ch >= N) ? (cnt += countd[ch], mass[i * THREADS3 + threadIdx.x])
                        : (cnt++, massd[ch]);
            // add child's contribution
            cm += m;
            px += posxd[ch] * m;
            py += posyd[ch] * m;
          }
        }

        countd[k] = cnt;
        const float m = 1.0f / cm;
        posxd[k] = px * m;
        posyd[k] = py * m;
        __threadfence();  // make sure data are visible before setting mass
        massd[k] = cm;
      }

      CONTINUE_LOOP:
      k += inc;  // move on to next cell
    }
    k = restart;
  }


  int j = 0;
  // iterate over all cells assigned to thread
  while (k <= NNODES)
  {
    if (massd[k] >= 0)
    {
      k += inc;
      goto SKIP_LOOP;
    }


    if (j == 0)
    {
      j = 4;
      for (int i = 0; i < 4; i++)
      {
        const int ch = childd[k * 4 + i];

        child[i * THREADS3 + threadIdx.x] = ch;
        if ((ch < N) or
           ((mass[i * THREADS3 + threadIdx.x] = massd[ch]) >= 0))
          j--;

      }
    }
    else
    {
      j = 4;
      for (int i = 0; i < 4; i++)
      {
        const int ch = child[i * THREADS3 + threadIdx.x];

        if ((ch < N) or
           (mass[i * THREADS3 + threadIdx.x] >= 0) or
           ((mass[i * THREADS3 + threadIdx.x] = massd[ch]) >= 0))
          j--;

      }
    }

    if (j == 0)
    {
      // all children are ready
      cm = 0.0f;
      px = 0.0f;
      py = 0.0f;
      int cnt = 0;

      #pragma unroll
      for (int i = 0; i < 4; i++)
      {
        const int ch = child[i * THREADS3 + threadIdx.x];
        if (ch >= 0)
        {
          const float m =
            (ch >= N) ? (cnt += countd[ch], mass[i * THREADS3 + threadIdx.x])
                      : (cnt++, massd[ch]);
          // add child's contribution
          cm += m;
          px += posxd[ch] * m;
          py += posyd[ch] * m;
        }
      }

      countd[k] = cnt;
      const float m = 1.0f / cm;
      posxd[k] = px * m;
      posyd[k] = py * m;
      flag = 1;
    }


    SKIP_LOOP:
    __syncthreads();
    if (flag != 0)
    {
      massd[k] = cm;
      k += inc;
      flag = 0;
    }
  }
}

/**
 * Sort the cells
 */
__global__ __launch_bounds__(THREADS4, FACTOR4) void
SortKernel(int *restrict sortd,
           const int *restrict countd,
           volatile int *restrict startd,
           int *restrict childd,
           const int NNODES,
           const int N,
           const int *restrict bottomd)
{
  const int bottom = bottomd[0];
  const int dec = blockDim.x * gridDim.x;
  int k = NNODES + 1 - dec + threadIdx.x + blockIdx.x * blockDim.x;
  int start;
  int limiter = 0;

  // iterate over all cells assigned to thread
  while (k >= bottom)
  {
    // To control possible infinite loops
    if (++limiter > NNODES)
      break;

    // Not a child so skip
    if ((start = startd[k]) < 0)
      continue;


    int j = 0;
    for (int i = 0; i < 4; i++)
    {
      const int ch = childd[k * 4 + i];
      if (ch >= 0)
      {
        if (i != j)
        {
          // move children to front (needed later for speed)
          childd[k * 4 + i] = -1;
          childd[k * 4 + j] = ch;
        }
        if (ch >= N)
        {
          // child is a cell
          startd[ch] = start;
          start += countd[ch];  // add #bodies in subtree
        }
        else if (start <= NNODES and start >= 0)
        {
          // child is a body
          sortd[start++] = ch;
        }
        j++;
      }
    }
    k -= dec;  // move on to next cell
  }
}


/**
 * Calculate the repulsive forces using the KD Tree
 */
__global__ __launch_bounds__(THREADS5, FACTOR5) void
RepulsionKernel( /* int *restrict errd, */
                const float theta,
                const float epssqd,  // correction for zero distance
                const int *restrict sortd,
                const int *restrict childd,
                const float *restrict massd,
                const float *restrict posxd,
                const float *restrict posyd,
                float *restrict velxd,
                float *restrict velyd,
                float *restrict Z_norm,
                const float theta_squared,
                const int NNODES,
                const int FOUR_NNODES,
                const int N,
                const float *restrict radiusd_squared,
                const int *restrict maxdepthd)
{
  // Return if max depth is too deep
  // Not possible since I limited it to 32
  // if (maxdepthd[0] > 32)
  // {
  //   atomicExch(errd, max_depth);
  //   return;
  // }

  __shared__ int pos[THREADS5], node[THREADS5];
  __shared__ float dq[THREADS5];

  if (threadIdx.x == 0)
  {
    const int max_depth = maxdepthd[0];
    dq[0] = __fdividef(radiusd_squared[0], theta_squared);

    for (int i = 1; i < max_depth; i++)
    {
      dq[i] = dq[i - 1] * 0.25f;
      dq[i - 1] += epssqd;
    }
    dq[max_depth - 1] += epssqd;
  }


  __syncthreads();
  // figure out first thread in each warp (lane 0)
  // const int base = threadIdx.x / 32;
  // const int sbase = base * 32;
  const int sbase = (threadIdx.x / 32) * 32;
  const bool SBASE_EQ_THREAD = (sbase == threadIdx.x);

  const int diff = threadIdx.x - sbase;
  // make multiple copies to avoid index calculations later
  // Always true
  // if (diff < 32)
  dq[diff + sbase] = dq[diff];


  //__syncthreads();
  __threadfence_block();

  // iterate over all bodies assigned to thread
  const int MAX_SIZE = FOUR_NNODES + 4;

  for (int k = threadIdx.x + blockIdx.x * blockDim.x; k < N; k += blockDim.x * gridDim.x)
  {
    const int i = sortd[k];  // get permuted/sorted index
    // cache position info
    if (i < 0 or i >= MAX_SIZE)
      continue;
    
    const float px = posxd[i];
    const float py = posyd[i];

    float vx = 0.0f;
    float vy = 0.0f;
    float normsum = 0.0f;

    // initialize iteration stack, i.e., push root node onto stack
    int depth = sbase;

    if (SBASE_EQ_THREAD == true)
    {
      pos[sbase] = 0;
      node[sbase] = FOUR_NNODES;
    }

    do {
      // stack is not empty
      int pd = pos[depth];
      int nd = node[depth];


      while (pd < 4)
      {
        const int index = nd + pd++;
        if (index < 0 or index >= MAX_SIZE)
          break;

        const int n = childd[index];  // load child pointer

        // Non child
        if (n < 0 or n > NNODES)
          break;

        const float dx = px - posxd[n];
        const float dy = py - posyd[n];
        const float dxy = dx*dx + dy*dy + epssqd; 


        if ((n < N) or __all_sync(__activemask(), dxy >= dq[depth]))
        {
          const float tdist = __fdividef(1.0f, (1.0f + dxy));
          float mult = massd[n] * tdist;
          normsum += mult;
          mult *= tdist;
          vx += dx * mult;
          vy += dy * mult;
        }
        else
        {
          // push cell onto stack
          if (SBASE_EQ_THREAD == true)
          {
            pos[depth] = pd;
            node[depth] = nd;
          }
          depth++;
          pd = 0;
          nd = n * 4;
        }
      }


    } while (--depth >= sbase); // done with this level


    // update velocity
    velxd[i] += vx;
    velyd[i] += vy;
    atomicAdd(Z_norm, normsum);
  }
}


/**
 * Find the norm(Y)
 */
__global__ void
get_norm(const float *restrict Y1,
         const float *restrict Y2,
         float *restrict norm,
         float *restrict norm_add1,
         const int N)
{
  const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= N) return;
  norm[i] = Y1[i] * Y1[i] + Y2[i] * Y2[i];
  norm_add1[i] = norm[i] + 1.0f;
}

/**
 * Fast attractive kernel. Uses COO matrix.
 */
__global__ void
attractive_kernel_bh(const float *restrict VAL,
                     const int *restrict COL,
                     const int *restrict ROW,
                     const float *restrict Y1,
                     const float *restrict Y2,
                     const float *restrict norm,
                     const float *restrict norm_add1,
                     float *restrict attract1,
                     float *restrict attract2,
                     const int NNZ)
{
  const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= NNZ) return;
  const int i = ROW[index];
  const int j = COL[index];

  // TODO: Calculate Kullback-Leibler divergence
  // TODO: Convert attractive forces to CSR format
  const float PQ = __fdividef(
    VAL[index],
    norm_add1[i] + norm[j] - 2.0f * (Y1[i] * Y1[j] + Y2[i] * Y2[j]));  // P*Q

  // Apply forces
  atomicAdd(&attract1[i], PQ * (Y1[i] - Y1[j]));
  atomicAdd(&attract2[i], PQ * (Y2[i] - Y2[j]));
}

/**
 * Apply gradient updates.
 */
__global__ __launch_bounds__(THREADS6, FACTOR6) void
IntegrationKernel(const float eta,
                  const float momentum,
                  const float exaggeration,
                  float *restrict Y1,
                  float *restrict Y2,
                  const float *restrict attract1,
                  const float *restrict attract2,
                  const float *restrict repel1,
                  const float *restrict repel2,
                  float *restrict gains1,
                  float *restrict gains2,
                  float *restrict old_forces1,
                  float *restrict old_forces2,
                  const float *restrict Z,
                  const int N)
{
  float ux, uy, gx, gy;

  // iterate over all bodies assigned to thread
  const int inc = blockDim.x * gridDim.x;
  const float Z_norm = Z[0];

  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += inc) {
    const float dx = attract1[i] - Z_norm * repel1[i];
    const float dy = attract2[i] - Z_norm * repel2[i];

    if (signbit(dx) != signbit(ux = old_forces1[i]))
      gx = gains1[i] + 0.2f;
    else
      gx = gains1[i] * 0.8f;
    if (gx < 0.01f) gx = 0.01f;

    if (signbit(dy) != signbit(uy = old_forces2[i]))
      gy = gains2[i] + 0.2f;
    else
      gy = gains2[i] * 0.8f;
    if (gy < 0.01f) gy = 0.01f;

    gains1[i] = gx;
    gains2[i] = gy;

    old_forces1[i] = ux = momentum * ux - eta * gx * dx;
    old_forces2[i] = uy = momentum * uy - eta * gy * dy;

    Y1[i] += ux;
    Y2[i] += uy;
  }
}

}  // namespace TSNE
}  // namespace ML
