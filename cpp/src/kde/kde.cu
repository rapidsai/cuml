/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/neighbors/kde.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuda/std/limits>

#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace ML::KDE {

// ============================================================================
// Distance accumulator ops — decomposed into init / accumulate / finalize
// so the tiled kernel can tile over features while accumulating partial
// distances in registers.
//
// Each specialisation defines:
//   N_ACC  — number of accumulator values per distance computation
//   init(acc)                — zero the accumulators
//   accumulate(acc, a, b, p) — per-feature accumulation
//   finalize(acc, d, p)      — convert accumulators to final scalar distance
// ============================================================================

template <typename T, ML::distance::DistanceType Metric>
struct DistOp;

// euclidean: sqrt(sum((a-b)^2))
template <typename T>
struct DistOp<T, ML::distance::DistanceType::L2SqrtUnexpanded> {
  static constexpr int N_ACC = 1;
  inline __device__ static void init(T* acc) { acc[0] = T(0); }
  inline __device__ static void accumulate(T* acc, T a, T b, T)
  {
    T d = a - b;
    acc[0] += d * d;
  }
  inline __device__ static T finalize(T* acc, int, T) { return sqrt(acc[0]); }
};

// sqeuclidean: sum((a-b)^2)
template <typename T>
struct DistOp<T, ML::distance::DistanceType::L2Expanded> {
  static constexpr int N_ACC = 1;
  inline __device__ static void init(T* acc) { acc[0] = T(0); }
  inline __device__ static void accumulate(T* acc, T a, T b, T)
  {
    T d = a - b;
    acc[0] += d * d;
  }
  inline __device__ static T finalize(T* acc, int, T) { return acc[0]; }
};

// manhattan: sum(|a-b|)
template <typename T>
struct DistOp<T, ML::distance::DistanceType::L1> {
  static constexpr int N_ACC = 1;
  inline __device__ static void init(T* acc) { acc[0] = T(0); }
  inline __device__ static void accumulate(T* acc, T a, T b, T) { acc[0] += abs(a - b); }
  inline __device__ static T finalize(T* acc, int, T) { return acc[0]; }
};

// chebyshev: max(|a-b|)
template <typename T>
struct DistOp<T, ML::distance::DistanceType::Linf> {
  static constexpr int N_ACC = 1;
  inline __device__ static void init(T* acc) { acc[0] = T(0); }
  inline __device__ static void accumulate(T* acc, T a, T b, T)
  {
    acc[0] = max(acc[0], abs(a - b));
  }
  inline __device__ static T finalize(T* acc, int, T) { return acc[0]; }
};

// minkowski: (sum(|a-b|^p))^(1/p)
template <typename T>
struct DistOp<T, ML::distance::DistanceType::LpUnexpanded> {
  static constexpr int N_ACC = 1;
  inline __device__ static void init(T* acc) { acc[0] = T(0); }
  inline __device__ static void accumulate(T* acc, T a, T b, T p) { acc[0] += pow(abs(a - b), p); }
  inline __device__ static T finalize(T* acc, int, T p) { return pow(acc[0], T(1) / p); }
};

// cosine: 1 - dot(a,b)/(||a||*||b||)
// acc[0]=dot, acc[1]=||a||^2, acc[2]=||b||^2
template <typename T>
struct DistOp<T, ML::distance::DistanceType::CosineExpanded> {
  static constexpr int N_ACC = 3;
  inline __device__ static void init(T* acc) { acc[0] = acc[1] = acc[2] = T(0); }
  inline __device__ static void accumulate(T* acc, T a, T b, T)
  {
    acc[0] += a * b;
    acc[1] += a * a;
    acc[2] += b * b;
  }
  inline __device__ static T finalize(T* acc, int, T)
  {
    T denom = sqrt(acc[1]) * sqrt(acc[2]);
    return (denom > T(0)) ? (T(1) - acc[0] / denom) : T(0);
  }
};

// correlation: cosine on mean-centred vectors (single-pass via sum identities)
// acc[0]=sum_a, acc[1]=sum_b, acc[2]=sum_a2, acc[3]=sum_b2, acc[4]=sum_ab
template <typename T>
struct DistOp<T, ML::distance::DistanceType::CorrelationExpanded> {
  static constexpr int N_ACC = 5;
  inline __device__ static void init(T* acc) { acc[0] = acc[1] = acc[2] = acc[3] = acc[4] = T(0); }
  inline __device__ static void accumulate(T* acc, T a, T b, T)
  {
    acc[0] += a;
    acc[1] += b;
    acc[2] += a * a;
    acc[3] += b * b;
    acc[4] += a * b;
  }
  inline __device__ static T finalize(T* acc, int d, T)
  {
    T ma  = acc[0] / T(d);
    T mb  = acc[1] / T(d);
    T dot = acc[4] - T(d) * ma * mb;
    T na  = acc[2] - T(d) * ma * ma;
    T nb  = acc[3] - T(d) * mb * mb;
    T den = sqrt(na) * sqrt(nb);
    return (den > T(0)) ? (T(1) - dot / den) : T(0);
  }
};

// canberra: sum(|a-b|/(|a|+|b|))
template <typename T>
struct DistOp<T, ML::distance::DistanceType::Canberra> {
  static constexpr int N_ACC = 1;
  inline __device__ static void init(T* acc) { acc[0] = T(0); }
  inline __device__ static void accumulate(T* acc, T a, T b, T)
  {
    const T diff = abs(a - b);
    const T add  = abs(a) + abs(b);
    acc[0] += ((add != T(0)) * diff / (add + (add == T(0))));
  }
  inline __device__ static T finalize(T* acc, int, T) { return acc[0]; }
};

// hellinger: sqrt(1 - sum(sqrt(a)*sqrt(b)))
template <typename T>
struct DistOp<T, ML::distance::DistanceType::HellingerExpanded> {
  static constexpr int N_ACC = 1;
  inline __device__ static void init(T* acc) { acc[0] = T(0); }
  inline __device__ static void accumulate(T* acc, T a, T b, T) { acc[0] += sqrt(a) * sqrt(b); }
  inline __device__ static T finalize(T* acc, int, T)
  {
    const T val = T(1) - acc[0];
    return sqrt((!signbit(val)) * val);
  }
};

// jensen-shannon
template <typename T>
struct DistOp<T, ML::distance::DistanceType::JensenShannon> {
  static constexpr int N_ACC = 1;
  inline __device__ static void init(T* acc) { acc[0] = T(0); }
  inline __device__ static void accumulate(T* acc, T a, T b, T)
  {
    const T m     = T(0.5) * (a + b);
    const bool mz = (m == T(0));
    const T logM  = (!mz) * log(m + mz);
    const bool xz = (a == T(0));
    const bool yz = (b == T(0));
    acc[0] += (-a * (logM - log(a + xz))) + (-b * (logM - log(b + yz)));
  }
  inline __device__ static T finalize(T* acc, int, T) { return sqrt(T(0.5) * acc[0]); }
};

// hamming: count(a!=b)/d
template <typename T>
struct DistOp<T, ML::distance::DistanceType::HammingUnexpanded> {
  static constexpr int N_ACC = 1;
  inline __device__ static void init(T* acc) { acc[0] = T(0); }
  inline __device__ static void accumulate(T* acc, T a, T b, T) { acc[0] += (a != b); }
  inline __device__ static T finalize(T* acc, int d, T) { return acc[0] / T(d); }
};

// KL divergence: sum(a*log(a/b))
template <typename T>
struct DistOp<T, ML::distance::DistanceType::KLDivergence> {
  static constexpr int N_ACC = 1;
  inline __device__ static void init(T* acc) { acc[0] = T(0); }
  inline __device__ static void accumulate(T* acc, T a, T b, T)
  {
    if (a > T(0) && b > T(0)) { acc[0] += a * log(a / b); }
  }
  inline __device__ static T finalize(T* acc, int, T) { return acc[0]; }
};

// Russell-Rao: (d - sum(a*b)) / d
template <typename T>
struct DistOp<T, ML::distance::DistanceType::RusselRaoExpanded> {
  static constexpr int N_ACC = 1;
  inline __device__ static void init(T* acc) { acc[0] = T(0); }
  inline __device__ static void accumulate(T* acc, T a, T b, T) { acc[0] += a * b; }
  inline __device__ static T finalize(T* acc, int d, T) { return (T(d) - acc[0]) / T(d); }
};

// ============================================================================
// Log-kernel traits — one specialisation per KernelType
// ============================================================================

template <typename T, KernelType K>
struct LogKernel;

template <typename T>
struct LogKernel<T, KernelType::Gaussian> {
  inline __device__ static T eval(T x, T h) { return -(x * x) / (T(2) * h * h); }
};

template <typename T>
struct LogKernel<T, KernelType::Tophat> {
  inline __device__ static T eval(T x, T h)
  {
    return (x < h) ? T(0) : cuda::std::numeric_limits<T>::lowest();
  }
};

template <typename T>
struct LogKernel<T, KernelType::Epanechnikov> {
  inline __device__ static T eval(T x, T h)
  {
    T z = max(T(1) - (x * x) / (h * h), T(1e-30));
    return (x < h) ? log(z) : cuda::std::numeric_limits<T>::lowest();
  }
};

template <typename T>
struct LogKernel<T, KernelType::Exponential> {
  inline __device__ static T eval(T x, T h) { return -x / h; }
};

template <typename T>
struct LogKernel<T, KernelType::Linear> {
  inline __device__ static T eval(T x, T h)
  {
    T z = max(T(1) - x / h, T(1e-30));
    return (x < h) ? log(z) : cuda::std::numeric_limits<T>::lowest();
  }
};

template <typename T>
struct LogKernel<T, KernelType::Cosine> {
  inline __device__ static T eval(T x, T h)
  {
    T z = max(cos(T(0.5) * T(M_PI) * x / h), T(1e-30));
    return (x < h) ? log(z) : cuda::std::numeric_limits<T>::lowest();
  }
};

// ============================================================================
// Host-side normalization functions (mirror the Python implementations)
// ============================================================================

template <typename T>
T logVn(int n)
{
  return T(0.5) * n * std::log(T(M_PI)) - std::lgamma(T(0.5) * n + T(1));
}

template <typename T>
T logSn(int n)
{
  return std::log(T(2) * T(M_PI)) + logVn<T>(n - 1);
}

template <typename T>
T norm_factor(KernelType kernel, T h, int d)
{
  T factor;
  switch (kernel) {
    case KernelType::Gaussian: factor = T(0.5) * d * std::log(T(2) * T(M_PI)); break;
    case KernelType::Tophat: factor = logVn<T>(d); break;
    case KernelType::Epanechnikov: factor = logVn<T>(d) + std::log(T(2) / T(d + 2)); break;
    case KernelType::Exponential: factor = logSn<T>(d - 1) + std::lgamma(T(d)); break;
    case KernelType::Linear: factor = logVn<T>(d) - std::log(T(d + 1)); break;
    case KernelType::Cosine: {
      T f   = T(0);
      T tmp = T(2) / T(M_PI);
      for (int k = 1; k <= d; k += 2) {
        f += tmp;
        tmp *= -(T(d - k) * T(d - k - 1)) * std::pow(T(2) / T(M_PI), 2);
      }
      factor = std::log(f) + logSn<T>(d - 1);
    } break;
    default: throw std::invalid_argument("Unsupported kernel type");
  }
  return factor + d * std::log(h);
}

// ============================================================================
// Tiled CUDA kernel — edistance-style CELL_TILE + FEAT_TILE optimisation
//
// One thread per query point.  Train vectors are cooperatively loaded into
// shared memory in tiles of [FEAT_TILE][CELL_TILE].  Each thread accumulates
// distances from its query point to CELL_TILE train points simultaneously,
// amortising query feature reads and reducing global memory traffic.
//
// Supports both single-pass (full train set → final output) and multi-pass
// (train subset → partial logsumexp) modes.  Multi-pass mode is used when
// the query count is too small to fill the GPU, parallelising over the
// train dimension via a 2D grid.
// ============================================================================

template <typename T, ML::distance::DistanceType Metric, KernelType Kernel, int CELL_TILE>
__global__ void kde_tiled_kernel(const T* __restrict__ query,
                                 const T* __restrict__ train,
                                 const T* __restrict__ weights,
                                 T* __restrict__ out_a,
                                 T* __restrict__ out_b,
                                 int n_query,
                                 int n_train,
                                 int d,
                                 T bandwidth,
                                 T metric_arg,
                                 T log_norm,
                                 int train_chunk,
                                 int feat_tile)
{
  using DOp = DistOp<T, Metric>;

  extern __shared__ char smem_raw[];
  T* smem_train = reinterpret_cast<T*>(smem_raw);  // [feat_tile][CELL_TILE]

  const int i      = blockIdx.x * blockDim.x + threadIdx.x;
  const bool valid = (i < n_query);

  constexpr int N_ACC = DOp::N_ACC;

  // Determine train range for this block
  const int j_begin = blockIdx.y * train_chunk;
  const int j_end   = min(j_begin + train_chunk, n_train);

  T running_max = -cuda::std::numeric_limits<T>::infinity();
  T running_sum = T(0);

  // Tile over train points in groups of CELL_TILE
  for (int j_base = j_begin; j_base < j_end; j_base += CELL_TILE) {
    const int cells_in_tile = min(CELL_TILE, j_end - j_base);

    // Per-train-point accumulators in registers
    T acc[CELL_TILE * N_ACC];
#pragma unroll
    for (int c = 0; c < CELL_TILE; ++c)
      DOp::init(&acc[c * N_ACC]);

    // Tile over features
    for (int feat_base = 0; feat_base < d; feat_base += feat_tile) {
      const int feats_in_tile = min(feat_tile, d - feat_base);

      // Cooperatively load train tile into shared memory: smem[feat][cell]
      const int total_elems = feat_tile * CELL_TILE;
      for (int idx = threadIdx.x; idx < total_elems; idx += blockDim.x) {
        const int cell = idx / feat_tile;
        const int feat = idx % feat_tile;
        T val          = T(0);
        if (cell < cells_in_tile && feat < feats_in_tile) {
          val = train[static_cast<size_t>(j_base + cell) * d + feat_base + feat];
        }
        smem_train[feat * CELL_TILE + cell] = val;
      }

      __syncthreads();

      if (valid) {
        for (int f = 0; f < feats_in_tile; ++f) {
          const T val_q = query[static_cast<size_t>(i) * d + feat_base + f];
#pragma unroll
          for (int c = 0; c < CELL_TILE; ++c) {
            const T val_t = smem_train[f * CELL_TILE + c];
            DOp::accumulate(&acc[c * N_ACC], val_q, val_t, metric_arg);
          }
        }
      }

      __syncthreads();
    }

    // Finalize distances and fold into streaming logsumexp
    if (valid) {
#pragma unroll
      for (int c = 0; c < CELL_TILE; ++c) {
        if (c >= cells_in_tile) break;
        T dist  = DOp::finalize(&acc[c * N_ACC], d, metric_arg);
        T log_k = LogKernel<T, Kernel>::eval(dist, bandwidth);
        if (weights) log_k += log(weights[j_base + c]);

        if (log_k > running_max) {
          running_sum = running_sum * exp(running_max - log_k) + T(1);
          running_max = log_k;
        } else {
          running_sum += exp(log_k - running_max);
        }
      }
    }
  }

  if (valid) {
    if (out_b == nullptr) {
      // Single-pass: write final log-probability
      out_a[i] = log(running_sum) + running_max - log_norm;
    } else {
      // Multi-pass: write partial (max, sum) for later reduction
      const size_t idx = static_cast<size_t>(i) * gridDim.y + blockIdx.y;
      out_a[idx]       = running_max;
      out_b[idx]       = running_sum;
    }
  }
}

// ============================================================================
// Reduction kernel — merges partial logsumexp results from multi-pass
// ============================================================================

template <typename T>
__global__ void kde_reduce_kernel(const T* __restrict__ partial_max,
                                  const T* __restrict__ partial_sum,
                                  T* __restrict__ output,
                                  int n_query,
                                  int n_blocks,
                                  T log_norm)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_query) return;

  T rmax = -cuda::std::numeric_limits<T>::infinity();
  T rsum = T(0);

  for (int b = 0; b < n_blocks; ++b) {
    const size_t idx = static_cast<size_t>(i) * n_blocks + b;
    const T pm       = partial_max[idx];
    const T ps       = partial_sum[idx];
    if (pm > rmax) {
      rsum = rsum * exp(rmax - pm) + ps;
      rmax = pm;
    } else {
      rsum += ps * exp(pm - rmax);
    }
  }
  output[i] = log(rsum) + rmax - log_norm;
}

// ============================================================================
// Double dispatch: runtime enum → compile-time template
// ============================================================================

template <typename T, typename Fn>
void dispatch_metric(ML::distance::DistanceType metric, Fn&& fn)
{
  using DT = ML::distance::DistanceType;
  switch (metric) {
    case DT::L2SqrtUnexpanded: fn(std::integral_constant<DT, DT::L2SqrtUnexpanded>{}); break;
    case DT::L2Expanded: fn(std::integral_constant<DT, DT::L2Expanded>{}); break;
    case DT::L1: fn(std::integral_constant<DT, DT::L1>{}); break;
    case DT::Linf: fn(std::integral_constant<DT, DT::Linf>{}); break;
    case DT::LpUnexpanded: fn(std::integral_constant<DT, DT::LpUnexpanded>{}); break;
    case DT::CosineExpanded: fn(std::integral_constant<DT, DT::CosineExpanded>{}); break;
    case DT::CorrelationExpanded: fn(std::integral_constant<DT, DT::CorrelationExpanded>{}); break;
    case DT::Canberra: fn(std::integral_constant<DT, DT::Canberra>{}); break;
    case DT::HellingerExpanded: fn(std::integral_constant<DT, DT::HellingerExpanded>{}); break;
    case DT::JensenShannon: fn(std::integral_constant<DT, DT::JensenShannon>{}); break;
    case DT::HammingUnexpanded: fn(std::integral_constant<DT, DT::HammingUnexpanded>{}); break;
    case DT::KLDivergence: fn(std::integral_constant<DT, DT::KLDivergence>{}); break;
    case DT::RusselRaoExpanded: fn(std::integral_constant<DT, DT::RusselRaoExpanded>{}); break;
    default: throw std::invalid_argument("Unsupported distance metric for KDE");
  }
}

template <typename T, typename Fn>
void dispatch_kernel(KernelType kernel, Fn&& fn)
{
  switch (kernel) {
    case KernelType::Gaussian:
      fn(std::integral_constant<KernelType, KernelType::Gaussian>{});
      break;
    case KernelType::Tophat: fn(std::integral_constant<KernelType, KernelType::Tophat>{}); break;
    case KernelType::Epanechnikov:
      fn(std::integral_constant<KernelType, KernelType::Epanechnikov>{});
      break;
    case KernelType::Exponential:
      fn(std::integral_constant<KernelType, KernelType::Exponential>{});
      break;
    case KernelType::Linear: fn(std::integral_constant<KernelType, KernelType::Linear>{}); break;
    case KernelType::Cosine: fn(std::integral_constant<KernelType, KernelType::Cosine>{}); break;
    default: throw std::invalid_argument("Unsupported kernel type for KDE");
  }
}

// ============================================================================
// Implementation: launches the tiled kernel (1-pass or 2-pass)
// ============================================================================

template <typename T>
void score_samples(const raft::handle_t& handle,
                   const T* query,
                   const T* train,
                   const T* weights,
                   T* output,
                   int n_query,
                   int n_train,
                   int d,
                   T bandwidth,
                   T sum_weights,
                   KernelType kernel,
                   ML::distance::DistanceType metric,
                   T metric_arg)
{
  cudaStream_t stream = handle.get_stream().value();
  T log_norm          = std::log(sum_weights) + norm_factor<T>(kernel, bandwidth, d);

  // Cap feature tile to the actual dimension to avoid wasted shared memory
  // and cooperative load cycles for low-dimensional data (e.g. 2D embeddings).
  const int feat_tile = min(64, d);
  // 512 threads for float32 (more cooperative load throughput, better GPU fill).
  // 256 for float64 to avoid exceeding per-block register limits with
  // CELL_TILE=64 double-precision accumulators (64×2 regs × 512 threads > 65536).
  const int threads  = (sizeof(T) == 4) ? 512 : 256;
  int n_query_blocks = (n_query + threads - 1) / threads;

  dispatch_metric<T>(metric, [&](auto metric_tag) {
    dispatch_kernel<T>(kernel, [&](auto kernel_tag) {
      constexpr auto M = decltype(metric_tag)::value;
      constexpr auto K = decltype(kernel_tag)::value;

      // Adapt CELL_TILE to keep accumulator register pressure under ~128 regs.
      constexpr int N_ACC     = DistOp<T, M>::N_ACC;
      constexpr int ACC_REGS  = sizeof(T) / 4;
      constexpr int RAW_TILE  = 128 / (N_ACC * ACC_REGS);
      constexpr int CELL_TILE = RAW_TILE >= 64   ? 64
                                : RAW_TILE >= 32 ? 32
                                : RAW_TILE >= 16 ? 16
                                : RAW_TILE >= 8  ? 8
                                                 : 4;

      size_t smem_bytes = feat_tile * CELL_TILE * sizeof(T);

      // Determine whether to split the train dimension across blocks.
      // When n_query is small the GPU is underutilised; splitting the train
      // set across a 2D grid exposes more parallelism.
      int sm_count = 0;
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
      int target_blocks   = sm_count * 4;
      int n_train_blocks  = max(1, target_blocks / n_query_blocks);
      int min_train_chunk = CELL_TILE * 4;
      n_train_blocks      = min(n_train_blocks, max(1, n_train / min_train_chunk));

      if (n_train_blocks <= 1) {
        // Single-pass: process all train points, write directly to output
        dim3 grid(n_query_blocks);
        kde_tiled_kernel<T, M, K, CELL_TILE>
          <<<grid, threads, smem_bytes, stream>>>(query,
                                                  train,
                                                  weights,
                                                  output,
                                                  static_cast<T*>(nullptr),
                                                  n_query,
                                                  n_train,
                                                  d,
                                                  bandwidth,
                                                  metric_arg,
                                                  log_norm,
                                                  n_train,
                                                  feat_tile);
        RAFT_CUDA_TRY(cudaPeekAtLastError());
      } else {
        // Multi-pass: split train dimension, write partial (max, sum), then reduce
        int train_chunk = (n_train + n_train_blocks - 1) / n_train_blocks;
        // Round up to CELL_TILE for clean tiling
        train_chunk = ((train_chunk + CELL_TILE - 1) / CELL_TILE) * CELL_TILE;
        // Recompute actual number of blocks after rounding
        n_train_blocks = (n_train + train_chunk - 1) / train_chunk;

        size_t buf_elems = static_cast<size_t>(n_query) * n_train_blocks;
        T* partial_max   = nullptr;
        T* partial_sum   = nullptr;
        RAFT_CUDA_TRY(cudaMallocAsync(&partial_max, buf_elems * sizeof(T), stream));
        RAFT_CUDA_TRY(cudaMallocAsync(&partial_sum, buf_elems * sizeof(T), stream));

        dim3 grid(n_query_blocks, n_train_blocks);
        kde_tiled_kernel<T, M, K, CELL_TILE><<<grid, threads, smem_bytes, stream>>>(query,
                                                                                    train,
                                                                                    weights,
                                                                                    partial_max,
                                                                                    partial_sum,
                                                                                    n_query,
                                                                                    n_train,
                                                                                    d,
                                                                                    bandwidth,
                                                                                    metric_arg,
                                                                                    log_norm,
                                                                                    train_chunk,
                                                                                    feat_tile);
        RAFT_CUDA_TRY(cudaPeekAtLastError());

        kde_reduce_kernel<T><<<n_query_blocks, threads, 0, stream>>>(
          partial_max, partial_sum, output, n_query, n_train_blocks, log_norm);
        RAFT_CUDA_TRY(cudaPeekAtLastError());

        RAFT_CUDA_TRY(cudaFreeAsync(partial_max, stream));
        RAFT_CUDA_TRY(cudaFreeAsync(partial_sum, stream));
      }
    });
  });
}

// Explicit instantiations
template void score_samples<float>(const raft::handle_t&,
                                   const float*,
                                   const float*,
                                   const float*,
                                   float*,
                                   int,
                                   int,
                                   int,
                                   float,
                                   float,
                                   KernelType,
                                   ML::distance::DistanceType,
                                   float);

template void score_samples<double>(const raft::handle_t&,
                                    const double*,
                                    const double*,
                                    const double*,
                                    double*,
                                    int,
                                    int,
                                    int,
                                    double,
                                    double,
                                    KernelType,
                                    ML::distance::DistanceType,
                                    double);

}  // namespace ML::KDE
