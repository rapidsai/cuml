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
#define restrict __restrict

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
 * Figures the bounding boxes for every point in the embedding.
 */
template <typename Index_t = int>
__global__ __launch_bounds__(THREADS1, FACTOR1) void
BoundingBoxKernel(int *restrict childd,           // FOUR_NNODES+4
                  const float *restrict posxd,    // NNODES+1
                  const float *restrict posyd,    // NNODES+1
                  float *restrict posxd_NNODES,
                  float *restrict posyd_NNODES,
                  float *restrict maxxd,    // blocks*FACTOR1 [80]
                  float *restrict maxyd,    // blocks*FACTOR1 [80]
                  float *restrict minxd,    // blocks*FACTOR1 [80]
                  float *restrict minyd,    // blocks*FACTOR1 [80]
                  const Index_t FOUR_NNODES,
                  const Index_t NNODES,
                  const Index_t N,
                  unsigned *restrict limiter,
                  float *restrict radiusd)
{
  float val, minx, maxx, miny, maxy;
  __shared__ float sminx[THREADS1], smaxx[THREADS1], sminy[THREADS1], smaxy[THREADS1];

  // initialize with valid data (in case #bodies < #threads)
  minx = maxx = posxd[0];
  miny = maxy = posyd[0];

  // scan all bodies
  const Index_t i = threadIdx.x;
  const Index_t inc = THREADS1 * gridDim.x;
  for (Index_t j = i + blockIdx.x * THREADS1; j < N; j += inc)
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

  for (Index_t j = THREADS1 / 2; j > i; j /= 2)
  {
    __syncthreads();
    const Index_t k = i + j;
    sminx[i] = minx = fminf(minx, sminx[k]);
    smaxx[i] = maxx = fmaxf(maxx, smaxx[k]);
    sminy[i] = miny = fminf(miny, sminy[k]);
    smaxy[i] = maxy = fmaxf(maxy, smaxy[k]);
  }
  

  if (i == 0)
  {
    // write block result to global memory
    const Index_t k = blockIdx.x;
    minxd[k] = minx;
    maxxd[k] = maxx;
    minyd[k] = miny;
    maxyd[k] = maxy;
    __threadfence();

    const Index_t inc = gridDim.x - 1;
    if (inc != atomicInc(limiter, inc)) return;

    // I'm the last block, so combine all block results
    for (Index_t j = 0; j <= inc; j++)
    {
      minx = fminf(minx, minxd[j]);
      maxx = fmaxf(maxx, maxxd[j]);
      miny = fminf(miny, minyd[j]);
      maxy = fmaxf(maxy, maxyd[j]);
    }

    // compute 'radius'
    atomicExch(radiusd, fmaxf(maxx - minx, maxy - miny) * 0.5f + 1e-5f);
    posxd_NNODES[0] = (minx + maxx) * 0.5f;
    posxd_NNODES[0] = (miny + maxy) * 0.5f;

    #pragma unroll
    for (Index_t a = 0; a < 4; a++)
      childd[FOUR_NNODES + a] = -1;
  }
}


/**
 * Clear some of the state vectors up.
 */
template <typename Index_t = int>
__global__ __launch_bounds__(1024, 1) void
ClearKernel1(int *restrict childd,
             const Index_t FOUR_NNODES,
             const Index_t FOUR_N)
{
  const Index_t inc = blockDim.x * gridDim.x;
  Index_t k = (FOUR_N & -32) + threadIdx.x + blockIdx.x * blockDim.x; 
  if (k < FOUR_N) k += inc;

  // iterate over all cells assigned to thread
  for (; k < FOUR_NNODES; k += inc) {
    childd[k] = -1;
  }
}


/**
 * Build the actual KD Tree.
 */
template <typename Index_t = int>
__global__ __launch_bounds__(THREADS2, FACTOR2) void
TreeBuildingKernel(int *restrict childd,        // (NNODES+1)*4
                   const float *restrict posxd, // NNODES+1
                   const float *restrict posyd, // NNODES+1
                   const Index_t NNODES,
                   const Index_t N,
                   int *restrict maxdepthd,
                   int *restrict bottomd,
                   const float *restrict radiusd)
{
  Index_t limiter = 0;
  Index_t j, depth;
  float x, y, r;
  float px, py;
  Index_t ch, n, locked, patch;

  // cache root data
  const float radius = radiusd[0];
  const float rootx = posxd[NNODES];
  const float rooty = posyd[NNODES];

  Index_t localmaxdepth = 1;
  Index_t skip = 1;
  const Index_t inc = blockDim.x * gridDim.x;
  Index_t i = threadIdx.x + blockIdx.x * blockDim.x;

  // iterate over all bodies assigned to thread
  while (i < N)
  {
    if (++limiter > NNODES) {
      break;
    }

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
      if (++depth > NNODES) {
        break;
      }
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
            // To control possible infinite loops
            if (++depth > NNODES) {
              break;
            }

            const Index_t cell = atomicSub(bottomd, 1) - 1;
            if (cell <= N) {
              // atomicExch(errd, 1);
              atomicExch(bottomd, NNODES);
            }

            if (patch != -1) {
              childd[n * 4 + j] = cell;
            }

            if (cell > patch)
              patch = cell;

            j = (x < posxd[ch]) ? 1 : 0;
            if (y < posyd[ch])
              j |= 2;

            childd[cell * 4 + j] = ch;
            n = cell;
            r *= 0.5f;

            x += (  (x < px) ?
                    (j = 1, r) : (j = 0, -r)  );

            y += (  (y < py) ?
                    (j |= 2, r) : (-r)  );

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
  if (localmaxdepth > 32)
    localmaxdepth = 32;

  atomicMax(maxdepthd, localmaxdepth);
}

/**
 * Clean more state vectors.
 */
template <typename Index_t = int>
__global__ __launch_bounds__(1024, 1) void
ClearKernel2(int *restrict startd,
             float *restrict massd,
             const Index_t NNODES,
             const int *restrict bottomd)
{
  const Index_t bottom = bottomd[0];
  const Index_t inc = blockDim.x * gridDim.x;
  Index_t k = (bottom & -32) + threadIdx.x + blockIdx.x * blockDim.x;
  if (k < bottom) k += inc;

  // iterate over all cells assigned to thread
  for (; k < NNODES; k += inc) {
    massd[k] = -1;
    startd[k] = -1;
  }
}

/**
 * Summarize the KD Tree via cell gathering
 */
template <typename Index_t = int>
__global__ __launch_bounds__(THREADS3, FACTOR3) void
SummarizationKernel(int *restrict countd,           // NNODES+1
                    const int *restrict childd,     // (NNODES+1)*4
                    volatile float *restrict massd, // NNODES+1
                    float *restrict posxd,          // NNODES+1
                    float *restrict posyd,          // NNODES+1
                    const Index_t NNODES,
                    const Index_t N,
                    const int *restrict bottomd) 
{
  Index_t limiter = 0;
  bool flag = 0;
  float cm, px, py;
  __shared__ int child[THREADS3 * 4];
  __shared__ float mass[THREADS3 * 4];

  const Index_t bottom = bottomd[0];
  const Index_t inc = blockDim.x * gridDim.x;
  Index_t k = (bottom & -32) + threadIdx.x + blockIdx.x * blockDim.x;
  if (k < bottom) k += inc;

  const Index_t restart = k;

  for (Index_t j = 0; j < 5; j++) // wait-free pre-passes
  {
    // iterate over all cells assigned to thread
    while (k <= NNODES)
    {
      if (massd[k] < 0)
      {
        for (Index_t i = 0; i < 4; i++)
        {
          const Index_t ch = childd[k * 4 + i];
          child[i * THREADS3 + threadIdx.x] = ch;

          if ((ch >= N) and ((mass[i * THREADS3 + threadIdx.x] = massd[ch]) < 0))
            goto CONTINUE_LOOP;
        }

        // all children are ready
        cm = 0;
        px = 0;
        py = 0;
        Index_t cnt = 0;

        #pragma unroll
        for (Index_t i = 0; i < 4; i++)
        {
          const Index_t ch = child[i * THREADS3 + threadIdx.x];
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


  Index_t j = 0;
  // iterate over all cells assigned to thread
  while (k <= NNODES)
  {
    if (++limiter > N)
      break;
    
    if (massd[k] >= 0)
    {
      k += inc;
      goto SKIP_LOOP;
    }


    if (j == 0)
    {
      j = 4;
      for (Index_t i = 0; i < 4; i++)
      {
        const Index_t ch = childd[k * 4 + i];

        child[i * THREADS3 + threadIdx.x] = ch;
        if (ch < N) {
          j--;
          continue;
        }

        if ((mass[i * THREADS3 + threadIdx.x] = massd[ch]) >= 0)
          j--;

      }
    }
    else
    {
      j = 4;
      for (Index_t i = 0; i < 4; i++)
      {
        const Index_t ch = child[i * THREADS3 + threadIdx.x];

        if ((ch < N) or
           (mass[i * THREADS3 + threadIdx.x] >= 0) or
           ((mass[i * THREADS3 + threadIdx.x] = massd[ch]) >= 0))
          j--;

      }
    }

    if (j == 0)
    {
      // all children are ready
      cm = 0;
      px = 0;
      py = 0;
      Index_t cnt = 0;

      #pragma unroll
      for (Index_t i = 0; i < 4; i++)
      {
        const Index_t ch = child[i * THREADS3 + threadIdx.x];
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
template <typename Index_t = int>
__global__ __launch_bounds__(THREADS4, FACTOR4) void
SortKernel(int *restrict sortd,             // NNODES+1
           const int *restrict countd,      // NNODES+1
           volatile int *restrict startd,   // NNODES+1
           int *restrict childd,            // (NNODES+1)*4
           const Index_t NNODES,
           const Index_t N,
           const int *restrict bottomd)
{
  const Index_t bottom = bottomd[0];
  const Index_t dec = blockDim.x * gridDim.x;
  Index_t k = NNODES + 1 - dec + threadIdx.x + blockIdx.x * blockDim.x;
  Index_t start;
  Index_t limiter = 0;

  // iterate over all cells assigned to thread
  while (k >= bottom)
  {
    // To control possible infinite loops
    if (++limiter > N)
      break;

    // Not a child so skip
    if ((start = startd[k]) < 0)
      continue;


    Index_t j = 0;
    for (Index_t i = 0; i < 4; i++)
    {
      const Index_t ch = childd[k * 4 + i];
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
template <typename Index_t = int>
__global__ __launch_bounds__(THREADS5, FACTOR5) void
RepulsionKernel(const float theta,
                const float epssqd,  // correction for zero distance
                const int *restrict sortd,        // NNODES+1
                const int *restrict childd,       // (NNODES+1)*4
                const float *restrict massd,      // NNODES+1
                const float *restrict posxd,      // NNODES+1
                const float *restrict posyd,      // NNODES+1
                float *restrict velxd,            // NNODES+1
                float *restrict velyd,            // NNODES+1
                float *restrict Z_norm,
                const float theta_squared,
                const Index_t NNODES,
                const Index_t FOUR_NNODES,
                const Index_t N,
                const float *restrict radiusd_squared,
                const int *restrict maxdepthd)
{
  Index_t limiter = 0;
  Index_t limiter2 = 0;
  const float EPS_PLUS_1 = epssqd + 1.0f;

  __shared__ int pos[THREADS5], node[THREADS5];
  __shared__ float dq[THREADS5];

  if (threadIdx.x == 0)
  {
    const Index_t max_depth = maxdepthd[0];
    dq[0] = __fdividef(radiusd_squared[0], theta_squared);

    for (Index_t i = 1; i < max_depth; i++)
    {
      dq[i] = dq[i - 1] * 0.25f;
      dq[i - 1] += epssqd;
    }
    dq[max_depth - 1] += epssqd;

    // Add one so EPS_PLUS_1 can be compared
    for (Index_t i = 0; i < max_depth; i++)
      dq[i] += 1.0f;
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

  __threadfence_block();

  // iterate over all bodies assigned to thread
  for (Index_t k = threadIdx.x + blockIdx.x * blockDim.x; k < N; k += blockDim.x * gridDim.x)
  {
    const Index_t i = sortd[k];  // get permuted/sorted index
    // cache position info
    if (i < 0 or i > NNODES)
      continue;
    
    const float px = posxd[i];
    const float py = posyd[i];

    float vx = 0.0f;
    float vy = 0.0f;
    float normsum = 0.0f;

    // initialize iteration stack, i.e., push root node onto stack
    Index_t depth = sbase;

    if (SBASE_EQ_THREAD == true)
    {
      pos[sbase] = 0;
      node[sbase] = FOUR_NNODES;
    }

    do {
      if (++limiter > N)
        break;

      // stack is not empty
      Index_t pd = pos[depth];
      Index_t nd = node[depth];

      while (pd < 4)
      {
        const Index_t index = nd + pd++;
        if (index < 0 or index >= FOUR_NNODES + 4 or ++limiter2 > NNODES)
          break;

        const Index_t n = childd[index];  // load child pointer

        // Non child
        if (n < 0 or n > NNODES)
          break;

        const float dx = px - posxd[n];
        const float dy = py - posyd[n];
        const float dxy1 = dx*dx + dy*dy + EPS_PLUS_1; 


        if ((n < N) or __all_sync(__activemask(), dxy1 >= dq[depth]))
        {
          const float tdist_2 = __fdividef(massd[n], dxy1 * dxy1);
          normsum += tdist_2 * dxy1;
          vx += dx * tdist_2;
          vy += dy * tdist_2;
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
template <typename Index_t = int>
__global__ void
get_norm(const float *restrict Y1,
         const float *restrict Y2,
         float *restrict norm,
         float *restrict norm_add1,
         const Index_t N)
{
  const Index_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= N) return;
  norm[i] = Y1[i] * Y1[i] + Y2[i] * Y2[i];
  norm_add1[i] = norm[i] + 1.0f;
}

/**
 * Fast attractive kernel. Uses COO matrix.
 */
template <typename Index_t = int>
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
                     const Index_t NNZ)
{
  const Index_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= NNZ) return;
  const Index_t i = ROW[index];
  const Index_t j = COL[index];

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
template <typename Index_t = int>
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
                  const Index_t N,
                  const float MAX_BOUNDS,
                  float *restrict sums)
{
  float ux, uy, gx, gy;

  // iterate over all bodies assigned to thread
  const Index_t inc = blockDim.x * gridDim.x;
  const float Z_norm = Z[0];

  for (Index_t i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += inc)
  {
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

    // Confirm Y1 and Y2 are within bounds (-MAX_BOUNDS, MAX_BOUNDS)
    // These are arbitrary but can help with outliers
    // Also reset both gains and old forces
    if (Y1[i] < -MAX_BOUNDS) {
      Y1[i] = -MAX_BOUNDS;
      gains1[i] = 1;
      old_forces1[i] = 0;
    }
    else if (Y1[i] > MAX_BOUNDS) {
      Y1[i] = MAX_BOUNDS;
      gains1[i] = 1;
      old_forces1[i] = 0;
    }
    if (Y2[i] < -MAX_BOUNDS) {
      Y2[i] = -MAX_BOUNDS;
      gains2[i] = 1;
      old_forces2[i] = 0;
    }
    else if (Y2[i] > MAX_BOUNDS) {
      Y2[i] = MAX_BOUNDS;
      gains2[i] = 1;
      old_forces2[i] = 0;
    }

    atomicAdd(&sums[0], Y1[i]);
    atomicAdd(&sums[1], Y2[i]);
  }
}


/**
 * Mean Centre and then add some noise
 */
template <typename Index_t = int>
__global__ void
mean_centre(float *restrict Y1,
            float *restrict Y2,
            const float *restrict means,
            const float N)
{
  const Index_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= N) return;
  Y1[i] -= means[0];
  Y2[i] -= means[1];

  if (i % 1000 == 0) {
    Y1[i] += 0.00001f;
    Y2[i] -= 0.00001f;
  }
}


}  // namespace TSNE
}  // namespace ML

#undef restrict
