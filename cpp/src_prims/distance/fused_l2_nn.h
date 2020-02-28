/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cuda_utils.h>
#include <stdint.h>
#include <cub/cub.cuh>
#include <limits>
#include <linalg/contractions.cuh>

namespace MLCommon {
namespace Distance {

template <typename LabelT, typename DataT>
struct KVPMinReduce {
  typedef cub::KeyValuePair<LabelT, DataT> KVP;

  DI KVP operator()(const KVP& a, const KVP& b) {
    return b.value < a.value ? b : a;
  }
};  // KVPMinReduce

template <typename LabelT, typename DataT>
struct MinAndDistanceReduceOp {
  typedef typename cub::KeyValuePair<LabelT, DataT> KVP;
  DI void operator()(KVP* out, const KVP& other) {
    if (other.value < out->value) {
      out->key = other.key;
      out->value = other.value;
    }
  }

  DI void init(KVP* out, DataT maxVal) {
    out->key = -1;
    out->value = maxVal;
  }
};

template <typename LabelT, typename DataT>
struct MinReduceOp {
  typedef typename cub::KeyValuePair<LabelT, DataT> KVP;
  DI void operator()(DataT* out, const KVP& other) {
    if (other.value < *out) {
      *out = other.value;
    }
  }

  DI void init(DataT* out, DataT maxVal) { *out = maxVal; }
};

template <typename DataT, typename OutT, typename IdxT, bool Sqrt,
          typename Policy, typename ReduceOpT,
          typename BaseClass = LinAlg::Contractions_NT<DataT, IdxT, Policy>>
struct FusedL2NN : public BaseClass {
 private:
  typedef Policy P;

  const DataT* xn;
  const DataT* yn;
  OutT* min;
  int* mutex;

  DataT *sxNorm, *syNorm;
  cub::KeyValuePair<IdxT, DataT>* sRed;

  DataT maxVal;

  DataT acc[P::AccRowsPerTh][P::AccColsPerTh];

  ReduceOpT redOp;

  static const DataT Two = (DataT)2.0;

 public:
  DI FusedL2NN(OutT* _min, const DataT* _x, const DataT* _y, const DataT* _xn,
               const DataT* _yn, IdxT _m, IdxT _n, IdxT _k, char* _smem,
               DataT _mv, int* _mut, ReduceOpT op)
    : BaseClass(_x, _y, _m, _n, _k, _smem),
      xn(_xn),
      yn(_yn),
      min(_min),
      mutex(_mut),
      sxNorm((DataT*)_smem),
      syNorm(&(sxNorm[P::Mblk])),
      sRed((cub::KeyValuePair<IdxT, DataT>*)_smem),
      maxVal(_mv),
      redOp(op) {}

  DI void run() {
    prolog();
    loop();
    __syncthreads();  // so that we can safely reuse smem
    epilog();
  }

 private:
  DI void prolog() {
    this->ldgsts(0);
    this->pageWr ^= 1;
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < P::AccColsPerTh; ++j) {
        acc[i][j] = BaseClass::Zero;
      }
    }
    __syncthreads();
  }

  DI void loop() {
    for (int kidx = P::Kblk; kidx < this->k; kidx += P::Kblk) {
      this->ldgsts(kidx);
      accumulate();  // on the previous k-block
      __syncthreads();
      this->pageWr ^= 1;
      this->pageRd ^= 1;
    }
    accumulate();  // last iteration
  }

  DI void epilog() {
    for (int i = threadIdx.x; i < P::Mblk; i += P::Nthreads) {
      auto idx = blockIdx.x * P::Mblk + i;
      sxNorm[i] = idx < this->m ? xn[idx] : maxVal;
    }
    for (int i = threadIdx.x; i < P::Nblk; i += P::Nthreads) {
      auto idx = blockIdx.y * P::Nblk + i;
      syNorm[i] = idx < this->n ? yn[idx] : maxVal;
    }
    __syncthreads();
    DataT regxn[P::AccRowsPerTh], regyn[P::AccColsPerTh];
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
      regxn[i] = sxNorm[i * P::AccThRows + this->accrowid];
    }
#pragma unroll
    for (int i = 0; i < P::AccColsPerTh; ++i) {
      regyn[i] = syNorm[i * P::AccThCols + this->acccolid];
    }
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < P::AccColsPerTh; ++j) {
        acc[i][j] = regxn[i] + regyn[j] - Two * acc[i][j];
      }
    }
    if (Sqrt) {
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
        for (int j = 0; j < P::AccColsPerTh; ++j) {
          acc[i][j] = mySqrt(acc[i][j]);
        }
      }
    }
    // reduce
    cub::KeyValuePair<IdxT, DataT> val[P::AccRowsPerTh];
    KVPMinReduce<IdxT, DataT> pairRedOp;
    auto lid = laneId();
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
      val[i] = {-1, maxVal};
#pragma unroll
      for (int j = 0; j < P::AccColsPerTh; ++j) {
        auto tmpkey = this->acccolid + j * P::AccThCols + blockIdx.y * P::Nblk;
        cub::KeyValuePair<IdxT, DataT> tmp = {tmpkey, acc[i][j]};
        if (tmpkey < this->n) val[i] = pairRedOp(tmp, val[i]);
      }
      __syncthreads();
#pragma unroll
      for (int j = P::AccThCols / 2; j > 0; j >>= 1) {
        auto tmpkey = shfl(val[i].key, lid + j);
        auto tmpvalue = shfl(val[i].value, lid + j);
        cub::KeyValuePair<IdxT, DataT> tmp = {tmpkey, tmpvalue};
        val[i] = pairRedOp(tmp, val[i]);
      }
    }
    if (lid % P::AccThCols == 0) {
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
        sRed[i * P::AccThCols + this->accrowid] = val[i];
      }
    }
    __syncthreads();
    updateResults();
  }

  DI void updateResults() {
    /**
     * @todo: From Volta onwards see if "coalesced" atomicCAS approach as
     *        written below helps improve perf
     * <code>
     *   auto tid = threadIdx.x;
     *   auto rid = IdxT(blockIdx.x) * P::Mblk + tid;
     *   if (rid < m) {
     *     auto val = sRed[i];
     *     while (atomicCAS(mutex + rid, 0, 1) == 1)
     *       ;
     *     __threadfence();
     *     redOp(min + rid, val);
     *     __threadfence();
     *     atomicCAS(mutex + rid, 1, 0);
     *   }
     * </code>
     */
    // for now have first lane from each warp update a unique output row. This
    // will resolve hang issues with pre-Volta architectures
    auto nWarps = blockDim.x / WarpSize;
    auto lid = laneId();
    auto ridx = IdxT(blockIdx.x) * P::Mblk;
    if (lid == 0) {
      for (int i = threadIdx.x / WarpSize; i < P::Mblk; i += nWarps) {
        auto rid = ridx + i;
        if (rid < this->m) {
          auto val = sRed[i];
          while (atomicCAS(mutex + rid, 0, 1) == 1)
            ;
          __threadfence();
          redOp(min + rid, val);
          __threadfence();
          atomicCAS(mutex + rid, 1, 0);
        }
      }
    }
  }

  DI void accumulate() {
#pragma unroll
    for (int ki = 0; ki < P::Kblk; ki += P::Veclen) {
      this->ldsXY(ki);
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
        for (int j = 0; j < P::AccColsPerTh; ++j) {
#pragma unroll
          for (int v = 0; v < P::Veclen; ++v) {
            acc[i][j] += this->regx[i][v] * this->regy[j][v];
          }
        }
      }
    }
  }
};  // struct FusedL2NN

template <typename DataT, typename OutT, typename IdxT, bool Sqrt,
          typename Policy, typename ReduceOpT>
__global__ __launch_bounds__(Policy::Nthreads, 2) void fusedL2NNkernel(
  OutT* min, const DataT* x, const DataT* y, const DataT* xn, const DataT* yn,
  IdxT m, IdxT n, IdxT k, DataT maxVal, int* mutex, ReduceOpT redOp) {
  extern __shared__ char smem[];
  FusedL2NN<DataT, OutT, IdxT, Sqrt, Policy, ReduceOpT> obj(
    min, x, y, xn, yn, m, n, k, smem, maxVal, mutex, redOp);
  obj.run();
}

template <typename DataT, typename OutT, typename IdxT, typename ReduceOpT>
__global__ void initKernel(OutT* min, IdxT m, DataT maxVal, ReduceOpT redOp) {
  auto tid = IdxT(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid < m) {
    redOp.init(min + tid, maxVal);
  }
}

template <typename DataT, typename OutT, typename IdxT, int VecLen,
          typename ReduceOpT>
void fusedL2NNImpl(OutT* min, const DataT* x, const DataT* y, const DataT* xn,
                   const DataT* yn, IdxT m, IdxT n, IdxT k, int* workspace,
                   ReduceOpT redOp, bool sqrt, bool initOutBuffer,
                   cudaStream_t stream) {
  typedef typename LinAlg::Policy4x4<DataT, VecLen>::Policy Policy;
  dim3 grid(ceildiv<int>(m, Policy::Mblk), ceildiv<int>(n, Policy::Nblk));
  dim3 blk(Policy::Nthreads);
  auto nblks = ceildiv<int>(m, Policy::Nthreads);
  auto maxVal = std::numeric_limits<DataT>::max();
  CUDA_CHECK(cudaMemsetAsync(workspace, 0, sizeof(int) * m, stream));
  if (initOutBuffer) {
    initKernel<DataT, OutT, IdxT, ReduceOpT>
      <<<nblks, Policy::Nthreads, 0, stream>>>(min, m, maxVal, redOp);
    CUDA_CHECK(cudaGetLastError());
  }
  if (sqrt) {
    fusedL2NNkernel<DataT, OutT, IdxT, true, Policy, ReduceOpT>
      <<<grid, blk, Policy::SmemSize, stream>>>(min, x, y, xn, yn, m, n, k,
                                                maxVal, workspace, redOp);
  } else {
    fusedL2NNkernel<DataT, OutT, IdxT, false, Policy, ReduceOpT>
      <<<grid, blk, Policy::SmemSize, stream>>>(min, x, y, xn, yn, m, n, k,
                                                maxVal, workspace, redOp);
  }
  CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Fused L2 distance and 1-nearest-neighbor computation in a single call.
 *
 * The benefits of such a call are 2-fold: 1) eliminate the need for an
 * intermediate buffer to store the output of gemm 2) reduce the memory read
 * traffic on this intermediate buffer, otherwise needed during the reduction
 * phase for 1-NN.
 *
 * @tparam DataT     data type
 * @tparam OutT      output type to either store 1-NN indices and their minimum
 *                   distances or store only the min distances. Accordingly, one
 *                   has to pass an appropriate `ReduceOpT`
 * @tparam IdxT      indexing arithmetic type
 * @tparam ReduceOpT A struct to perform the final needed reduction operation
 *                   and also to initialize the output array elements with the
 *                   appropriate initial value needed for reduction.
 *
 * @param[out] min           will contain the reduced output (Length = `m`)
 *                           (on device)
 * @param[in]  x             first matrix. Row major. Dim = `m x k`.
 *                           (on device).
 * @param[in]  y             second matrix. Row major. Dim = `n x k`.
 *                           (on device).
 * @param[in]  xn            L2 squared norm of `x`. Length = `m`. (on device).
 * @param[in]  yn            L2 squared norm of `y`. Length = `n`. (on device)
 * @param[in]  m             gemm m
 * @param[in]  n             gemm n
 * @param[in]  k             gemm k
 * @param[in]  workspace     temp workspace. Size = sizeof(int)*m. (on device)
 * @param[in]  sqrt          Whether the output `minDist` should contain L2-sqrt
 * @param[in]  initOutBuffer whether to initialize the output buffer before the
 *                           main kernel launch
 * @param[in]  stream        cuda stream
 */
template <typename DataT, typename OutT, typename IdxT, typename ReduceOpT>
void fusedL2NN(OutT* min, const DataT* x, const DataT* y, const DataT* xn,
               const DataT* yn, IdxT m, IdxT n, IdxT k, void* workspace,
               ReduceOpT redOp, bool sqrt, bool initOutBuffer,
               cudaStream_t stream) {
  size_t bytes = sizeof(DataT) * k;
  if (16 % sizeof(DataT) == 0 && bytes % 16 == 0) {
    fusedL2NNImpl<DataT, OutT, IdxT, 16 / sizeof(DataT), ReduceOpT>(
      min, x, y, xn, yn, m, n, k, (int*)workspace, redOp, sqrt, initOutBuffer,
      stream);
  } else if (8 % sizeof(DataT) == 0 && bytes % 8 == 0) {
    fusedL2NNImpl<DataT, OutT, IdxT, 8 / sizeof(DataT), ReduceOpT>(
      min, x, y, xn, yn, m, n, k, (int*)workspace, redOp, sqrt, initOutBuffer,
      stream);
  } else {
    fusedL2NNImpl<DataT, OutT, IdxT, 1, ReduceOpT>(min, x, y, xn, yn, m, n, k,
                                                   (int*)workspace, redOp, sqrt,
                                                   initOutBuffer, stream);
  }
}

}  // namespace Distance
}  // namespace MLCommon
