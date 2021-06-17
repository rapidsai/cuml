/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cub/cub.cuh>

#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>

/// TODO: move to raft, refactor

namespace MLCommon {
namespace LinAlg {

/**
 * @todo: docs
 */
template <int _kblk, int _rpt, int _cpt, int _tr, int _tc>
struct BlockGemmPolicy {
  static constexpr int RowsPerTh = _rpt;
  static constexpr int ColsPerTh = _cpt;
  static constexpr int WorkPerTh = RowsPerTh * ColsPerTh;
  static constexpr int ThRows = _tr;
  static constexpr int ThCols = _tc;
  static constexpr int Kblk = _kblk;
  static constexpr int Mblk = RowsPerTh * ThRows;
  static constexpr int Nblk = ColsPerTh * ThCols;
  static constexpr int BlockSize = ThRows * ThCols;
};  // struct BlockGemmPolicy

/**
 * @todo: docs
 */
template <int _tr, int _tc>
struct BlockGemvPolicy {
  static constexpr int ThRows = _tr;
  static constexpr int ThCols = _tc;
  static constexpr int BlockSize = ThRows * ThCols;
};  // struct BlockGemmPolicy

template <typename GemmPolicy, typename T>
struct GemmStorage {
  T a_tile[GemmPolicy::Mblk * GemmPolicy::Kblk];
  T b_tile[GemmPolicy::Nblk * GemmPolicy::Kblk];
};

template <typename GemvPolicy, typename T>
struct GemvStorage {
  T acc[GemvPolicy::BlockSize];
};

template <int BlockSize, typename T>
struct ReductionStorage {
  using BlockReduce =
    cub::BlockReduce<T, BlockSize, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;

  typename BlockReduce::TempStorage temp;
  T broadcast;
};

/// TODO: more efficient implementation, vectorization
template <typename GemmPolicy, typename T>
DI void _block_gemm_load_tile(const T* global_matrix, T* shared_tile, int i0,
                              int j0, int m, int n, int tm, int tn) {
  for (int idx = threadIdx.x; idx < tm * tn; idx += GemmPolicy::BlockSize) {
    int ti = idx % tm;
    int tj = idx / tm;
    int i = i0 + ti;
    int j = j0 + tj;
    shared_tile[tj * tm + ti] =
      (i < m && j < n) ? global_matrix[j * m + i] : (T)0;
  }
}

/// TODO: more efficient implementation, vectorization
template <typename GemmPolicy, typename T>
DI void _block_gemm_load_transpose_tile(const T* global_matrix, T* shared_tile,
                                        int i0, int j0, int m, int n, int tm,
                                        int tn) {
  for (int idx = threadIdx.x; idx < tm * tn; idx += GemmPolicy::BlockSize) {
    int ti = idx / tn;
    int tj = idx % tn;
    int i = i0 + ti;
    int j = j0 + tj;
    shared_tile[tj * tm + ti] =
      (i < m && j < n) ? global_matrix[i * n + j] : (T)0;
  }
}

/**
 * @todo: docs
 * @note: no beta arg, 0 assumed
 */
template <typename GemmPolicy, typename T, typename StorageT>
DI void _block_gemm(bool transa, bool transb, int m, int n, int k, T alpha,
                    const T* a, const T* b, T* c, StorageT& gemm_storage) {
  /// TODO: more efficient implementation!
  ///       Can base it on raft/linalg/contractions.cuh

  const int th_off_i = threadIdx.x % GemmPolicy::ThRows;
  const int th_off_j = threadIdx.x / GemmPolicy::ThRows;

  T* shared_a_tile = gemm_storage.a_tile;
  T* shared_b_tile = gemm_storage.b_tile;
  T reg_acc[GemmPolicy::WorkPerTh];

  /* Loop over blocks of C */
  for (int blk_j = 0; blk_j < raft::ceildiv<int>(n, GemmPolicy::Nblk);
       blk_j++) {
    for (int blk_i = 0; blk_i < raft::ceildiv<int>(m, GemmPolicy::Mblk);
         blk_i++) {
      /* Initialize accumulation registers */
      for (int i = 0; i < GemmPolicy::WorkPerTh; i++) {
        reg_acc[i] = (T)0;
      }

      /* Loop over tiles in A and B corresponding to that block in C */
      for (int tile_k = 0; tile_k < raft::ceildiv<int>(k, GemmPolicy::Kblk);
           tile_k++) {
        /* Load a tile from A */
        if (transa)
          _block_gemm_load_transpose_tile<GemmPolicy>(
            a, shared_a_tile, blk_i * GemmPolicy::Mblk,
            tile_k * GemmPolicy::Kblk, m, k, GemmPolicy::Mblk,
            GemmPolicy::Kblk);
        else
          _block_gemm_load_tile<GemmPolicy>(a, shared_a_tile,
                                            blk_i * GemmPolicy::Mblk,
                                            tile_k * GemmPolicy::Kblk, m, k,
                                            GemmPolicy::Mblk, GemmPolicy::Kblk);
        /* Load a tile from B */
        if (transb)
          _block_gemm_load_transpose_tile<GemmPolicy>(
            b, shared_b_tile, tile_k * GemmPolicy::Kblk,
            blk_j * GemmPolicy::Nblk, k, n, GemmPolicy::Kblk, GemmPolicy::Nblk);
        else
          _block_gemm_load_tile<GemmPolicy>(
            b, shared_b_tile, tile_k * GemmPolicy::Kblk,
            blk_j * GemmPolicy::Nblk, k, n, GemmPolicy::Kblk, GemmPolicy::Nblk);

        __syncthreads();

        /* Loop over accumulators owned by this thread */
#pragma unroll
        for (int th_j = 0; th_j < GemmPolicy::ColsPerTh; th_j++) {
#pragma unroll
          for (int th_i = 0; th_i < GemmPolicy::RowsPerTh; th_i++) {
            int i = th_off_i + th_i * GemmPolicy::ThRows;
            int j = th_off_j + th_j * GemmPolicy::ThCols;
            /* Loop over corresponding items in the tile and accumulate */
#pragma unroll
            for (int th_k = 0; th_k < GemmPolicy::Kblk; th_k++) {
              reg_acc[GemmPolicy::RowsPerTh * th_j + th_i] +=
                shared_a_tile[GemmPolicy::Mblk * th_k + i] *
                shared_b_tile[GemmPolicy::Kblk * j + th_k];
            }
          }
        }

        __syncthreads();
      }

      /* Write accumulators in C */
#pragma unroll
      for (int th_j = 0; th_j < GemmPolicy::ColsPerTh; th_j++) {
#pragma unroll
        for (int th_i = 0; th_i < GemmPolicy::RowsPerTh; th_i++) {
          int i =
            blk_i * GemmPolicy::Mblk + th_off_i + th_i * GemmPolicy::ThRows;
          int j =
            blk_j * GemmPolicy::Nblk + th_off_j + th_j * GemmPolicy::ThCols;
          if (i < m and j < n)
            c[j * m + i] = alpha * reg_acc[th_j * GemmPolicy::RowsPerTh + th_i];
        }
      }
    }
  }
}

/**
 * @todo: docs
 * @note: no beta arg, 0 assumed
 */
template <typename GemvPolicy, typename T, typename StorageT>
DI void _block_gemv(int m, int n, T alpha, const T* a, const T* x, T* y,
                    StorageT& gemv_storage, T* shared_vec) {
  /// TODO: more efficient implementation

  /* Load x into shared vector */
  for (int i = threadIdx.x; i < n; i += GemvPolicy::BlockSize) {
    shared_vec[i] = x[i];
  }
  __syncthreads();

  const int th_off_i = threadIdx.x % GemvPolicy::ThRows;
  const int th_off_j = threadIdx.x / GemvPolicy::ThRows;

  /* Loop on tiles */
  for (int tile_i = 0; tile_i < raft::ceildiv<int>(m, GemvPolicy::ThRows);
       tile_i++) {
    int i = tile_i * GemvPolicy::ThRows + th_off_i;
    if (i < m) {
      /* Accumulate values owned by this thread and write to shared mem */
      T acc = 0;
      for (int j = th_off_j; j < n; j += GemvPolicy::ThCols) {
        acc += a[j * m + i] * shared_vec[j];
      }

      gemv_storage.acc[threadIdx.x] = acc;
    }
    __syncthreads();
    if (i < m) {
      /* First thread in each row does sequential reduction of other's results */
      T acc = 0;
      if (th_off_j == 0) {
        for (int tj = 0; tj < GemvPolicy::ThCols; tj++) {
          acc += gemv_storage.acc[tj * GemvPolicy::ThRows + th_off_i];
        }
        y[i] = alpha * acc;
      }
    }
  }
}

/** y = alpha * x */
template <int BlockSize, typename T>
DI void _block_ax(int n, T alpha, const T* x, T* y) {
  for (int i = threadIdx.x; i < n; i += BlockSize) {
    y[i] = alpha * x[i];
  }
}

template <int BlockSize, bool Broadcast, typename T, typename StorageT>
DI T _block_reduce(T& val, StorageT& reduction_storage) {
  using BlockReduce =
    cub::BlockReduce<T, BlockSize, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;

  T dot = BlockReduce(reduction_storage.temp).Sum(val);

  if (Broadcast) {
    if (threadIdx.x == 0) reduction_storage.broadcast = dot;
    __syncthreads();
    dot = reduction_storage.broadcast;
  }

  return dot;
}

template <int BlockSize, bool Broadcast, typename T, typename StorageT>
DI T _block_dot(int n, const T* x, const T* y, StorageT& reduction_storage) {
  /* Compute dot product terms and sequential reduction per thread */
  T acc = (T)0;
  for (int i = threadIdx.x; i < n; i += BlockSize) {
    acc += x[i] * y[i];
  }

  /* Complete reduction and return dot product */
  return _block_reduce<BlockSize, Broadcast>(acc, reduction_storage);
}

template <int BlockSize, bool Broadcast, typename T, typename StorageT>
DI T _block_xAxt(int n, const T* x, const T* A, StorageT& reduction_storage,
                 T* shared_vec) {
  /* Load x into shared vector */
  for (int i = threadIdx.x; i < n; i += BlockSize) {
    shared_vec[i] = x[i];
  }
  __syncthreads();

  /* Compute terms and sequential reduction per thread */
  T acc = (T)0;
  for (int idx = threadIdx.x; idx < n * n; idx += BlockSize) {
    int i = idx % n;
    int j = idx / n;

    acc += shared_vec[i] * A[idx] * shared_vec[j];
  }

  /* Complete reduction and return dot product */
  return _block_reduce<BlockSize, Broadcast>(acc, reduction_storage);
}

}  // namespace LinAlg
}  // namespace MLCommon
