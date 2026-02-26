/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/neighbors/kde.hpp>

#include <raft/core/handle.hpp>

#include <cuda/std/limits>

#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace ML::KDE {

// ============================================================================
// Distance traits — one partial specialisation per DistanceType.
//
// These mirror the per-element math in RAFT's distance ops
// (raft::distance::detail::ops), but are self-contained because:
// 1. RAFT distance op headers are internal detail headers not installed
//    with the public RAFT package.
// 2. RAFT's tiled API (core + epilog on 2D accumulator tiles) doesn't
//    map to our one-thread-per-query streaming pattern.
// 3. Norms-based metrics (cosine, correlation) need inline computation
//    since RAFT expects precomputed norms + shared memory.
// ============================================================================

template <typename T, ML::distance::DistanceType Metric>
struct Distance;

// euclidean: sqrt(sum((a-b)^2))
template <typename T>
struct Distance<T, ML::distance::DistanceType::L2SqrtUnexpanded> {
  __device__ static T compute(const T* a, const T* b, int d, T)
  {
    T acc = T(0);
    for (int i = 0; i < d; ++i) {
      const T diff = a[i] - b[i];
      acc += diff * diff;
    }
    return sqrt(acc);
  }
};

// sqeuclidean: sum((a-b)^2)
template <typename T>
struct Distance<T, ML::distance::DistanceType::L2Expanded> {
  __device__ static T compute(const T* a, const T* b, int d, T)
  {
    T acc = T(0);
    for (int i = 0; i < d; ++i) {
      const T diff = a[i] - b[i];
      acc += diff * diff;
    }
    return acc;
  }
};

// manhattan: sum(|a-b|)
template <typename T>
struct Distance<T, ML::distance::DistanceType::L1> {
  __device__ static T compute(const T* a, const T* b, int d, T)
  {
    T acc = T(0);
    for (int i = 0; i < d; ++i) {
      acc += abs(a[i] - b[i]);
    }
    return acc;
  }
};

// chebyshev: max(|a-b|)
template <typename T>
struct Distance<T, ML::distance::DistanceType::Linf> {
  __device__ static T compute(const T* a, const T* b, int d, T)
  {
    T mx = T(0);
    for (int i = 0; i < d; ++i) {
      mx = max(mx, abs(a[i] - b[i]));
    }
    return mx;
  }
};

// minkowski: (sum(|a-b|^p))^(1/p)
template <typename T>
struct Distance<T, ML::distance::DistanceType::LpUnexpanded> {
  __device__ static T compute(const T* a, const T* b, int d, T p)
  {
    T acc = T(0);
    for (int i = 0; i < d; ++i) {
      acc += pow(abs(a[i] - b[i]), p);
    }
    return pow(acc, T(1) / p);
  }
};

// cosine: 1 - dot(a,b)/(||a||*||b||)
template <typename T>
struct Distance<T, ML::distance::DistanceType::CosineExpanded> {
  __device__ static T compute(const T* a, const T* b, int d, T)
  {
    T dot = T(0), na = T(0), nb = T(0);
    for (int i = 0; i < d; ++i) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    T denom = sqrt(na) * sqrt(nb);
    return (denom > T(0)) ? (T(1) - dot / denom) : T(0);
  }
};

// correlation: cosine on mean-centred vectors
template <typename T>
struct Distance<T, ML::distance::DistanceType::CorrelationExpanded> {
  __device__ static T compute(const T* a, const T* b, int d, T)
  {
    T ma = T(0), mb = T(0);
    for (int i = 0; i < d; ++i) {
      ma += a[i];
      mb += b[i];
    }
    ma /= T(d);
    mb /= T(d);

    T dot = T(0), na = T(0), nb = T(0);
    for (int i = 0; i < d; ++i) {
      T da = a[i] - ma;
      T db = b[i] - mb;
      dot += da * db;
      na += da * da;
      nb += db * db;
    }
    T denom = sqrt(na) * sqrt(nb);
    return (denom > T(0)) ? (T(1) - dot / denom) : T(0);
  }
};

// canberra: sum(|a-b|/(|a|+|b|))
// Uses branchless formulation matching RAFT's canberra_distance_op
template <typename T>
struct Distance<T, ML::distance::DistanceType::Canberra> {
  __device__ static T compute(const T* a, const T* b, int d, T)
  {
    T acc = T(0);
    for (int i = 0; i < d; ++i) {
      const T diff = abs(a[i] - b[i]);
      const T add  = abs(a[i]) + abs(b[i]);
      acc += ((add != T(0)) * diff / (add + (add == T(0))));
    }
    return acc;
  }
};

// hellinger: sqrt(1 - sum(x*y))
// Matches RAFT's hellinger_distance_op: core accumulates x*y, epilog
// applies sqrt(max(0, 1-acc)) using signbit for branchless NaN avoidance.
template <typename T>
struct Distance<T, ML::distance::DistanceType::HellingerExpanded> {
  __device__ static T compute(const T* a, const T* b, int d, T)
  {
    T acc = T(0);
    for (int i = 0; i < d; ++i) {
      acc += a[i] * b[i];
    }
    const T val = T(1) - acc;
    return sqrt((!signbit(val)) * val);
  }
};

// jensen-shannon: sqrt(0.5 * sum(-x*(log(m)-log(x)) + -y*(log(m)-log(y))))
// Matches RAFT's jensen_shannon_distance_op
template <typename T>
struct Distance<T, ML::distance::DistanceType::JensenShannon> {
  __device__ static T compute(const T* a, const T* b, int d, T)
  {
    T acc = T(0);
    for (int i = 0; i < d; ++i) {
      const T m     = T(0.5) * (a[i] + b[i]);
      const bool mz = (m == T(0));
      const T logM  = (!mz) * log(m + mz);
      const bool xz = (a[i] == T(0));
      const bool yz = (b[i] == T(0));
      acc += (-a[i] * (logM - log(a[i] + xz))) + (-b[i] * (logM - log(b[i] + yz)));
    }
    return sqrt(T(0.5) * acc);
  }
};

// hamming: count(a!=b)/d
template <typename T>
struct Distance<T, ML::distance::DistanceType::HammingUnexpanded> {
  __device__ static T compute(const T* a, const T* b, int d, T)
  {
    T acc = T(0);
    for (int i = 0; i < d; ++i) {
      acc += (a[i] != b[i]);
    }
    return acc / T(d);
  }
};

// KL divergence: sum(a*log(a/b))
template <typename T>
struct Distance<T, ML::distance::DistanceType::KLDivergence> {
  __device__ static T compute(const T* a, const T* b, int d, T)
  {
    T acc = T(0);
    for (int i = 0; i < d; ++i) {
      if (a[i] > T(0) && b[i] > T(0)) { acc += a[i] * log(a[i] / b[i]); }
    }
    return acc;
  }
};

// Russell-Rao: (d - sum(a*b)) / d
// Matches RAFT's russel_rao_distance_op
template <typename T>
struct Distance<T, ML::distance::DistanceType::RusselRaoExpanded> {
  __device__ static T compute(const T* a, const T* b, int d, T)
  {
    T acc = T(0);
    for (int i = 0; i < d; ++i) {
      acc += a[i] * b[i];
    }
    return (T(d) - acc) / T(d);
  }
};

// ============================================================================
// Log-kernel traits — one specialisation per KernelType
// ============================================================================

template <typename T, KernelType K>
struct LogKernel;

template <typename T>
struct LogKernel<T, KernelType::Gaussian> {
  __device__ static T eval(T x, T h) { return -(x * x) / (T(2) * h * h); }
};

template <typename T>
struct LogKernel<T, KernelType::Tophat> {
  __device__ static T eval(T x, T h)
  {
    return (x < h) ? T(0) : cuda::std::numeric_limits<T>::lowest();
  }
};

template <typename T>
struct LogKernel<T, KernelType::Epanechnikov> {
  __device__ static T eval(T x, T h)
  {
    T z = max(T(1) - (x * x) / (h * h), T(1e-30));
    return (x < h) ? log(z) : cuda::std::numeric_limits<T>::lowest();
  }
};

template <typename T>
struct LogKernel<T, KernelType::Exponential> {
  __device__ static T eval(T x, T h) { return -x / h; }
};

template <typename T>
struct LogKernel<T, KernelType::Linear> {
  __device__ static T eval(T x, T h)
  {
    T z = max(T(1) - x / h, T(1e-30));
    return (x < h) ? log(z) : cuda::std::numeric_limits<T>::lowest();
  }
};

template <typename T>
struct LogKernel<T, KernelType::Cosine> {
  __device__ static T eval(T x, T h)
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
// Fused CUDA kernel — one thread per query point, streaming over training set
// ============================================================================

template <typename T, ML::distance::DistanceType Metric, KernelType Kernel>
__global__ void kde_fused_kernel(const T* __restrict__ query,
                                 const T* __restrict__ train,
                                 const T* __restrict__ weights,
                                 T* __restrict__ output,
                                 int n_query,
                                 int n_train,
                                 int d,
                                 T bandwidth,
                                 T metric_arg,
                                 T log_norm)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_query) return;

  T running_max = -cuda::std::numeric_limits<T>::infinity();
  T running_sum = T(0);

  for (int j = 0; j < n_train; ++j) {
    T dist  = Distance<T, Metric>::compute(&query[i * d], &train[j * d], d, metric_arg);
    T log_k = LogKernel<T, Kernel>::eval(dist, bandwidth);
    if (weights) log_k += log(weights[j]);

    // streaming logsumexp
    if (log_k > running_max) {
      running_sum = running_sum * exp(running_max - log_k) + T(1);
      running_max = log_k;
    } else {
      running_sum += exp(log_k - running_max);
    }
  }
  output[i] = log(running_sum) + running_max - log_norm;
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
// Implementation: launches the fused kernel
// ============================================================================

template <typename T>
void score_samples_impl(cudaStream_t stream,
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
  T log_norm = std::log(sum_weights) + norm_factor<T>(kernel, bandwidth, d);

  constexpr int threads = 256;
  int blocks            = (n_query + threads - 1) / threads;

  dispatch_metric<T>(metric, [&](auto metric_tag) {
    dispatch_kernel<T>(kernel, [&](auto kernel_tag) {
      constexpr auto M = decltype(metric_tag)::value;
      constexpr auto K = decltype(kernel_tag)::value;
      kde_fused_kernel<T, M, K><<<blocks, threads, 0, stream>>>(
        query, train, weights, output, n_query, n_train, d, bandwidth, metric_arg, log_norm);
    });
  });
}

// ============================================================================
// Public API (float + double overloads)
// ============================================================================

void score_samples(const raft::handle_t& handle,
                   const float* query,
                   const float* train,
                   const float* weights,
                   float* output,
                   int n_query,
                   int n_train,
                   int n_features,
                   float bandwidth,
                   float sum_weights,
                   KernelType kernel,
                   ML::distance::DistanceType metric,
                   float metric_arg)
{
  score_samples_impl<float>(handle.get_stream().value(),
                            query,
                            train,
                            weights,
                            output,
                            n_query,
                            n_train,
                            n_features,
                            bandwidth,
                            sum_weights,
                            kernel,
                            metric,
                            metric_arg);
}

void score_samples(const raft::handle_t& handle,
                   const double* query,
                   const double* train,
                   const double* weights,
                   double* output,
                   int n_query,
                   int n_train,
                   int n_features,
                   double bandwidth,
                   double sum_weights,
                   KernelType kernel,
                   ML::distance::DistanceType metric,
                   double metric_arg)
{
  score_samples_impl<double>(handle.get_stream().value(),
                             query,
                             train,
                             weights,
                             output,
                             n_query,
                             n_train,
                             n_features,
                             bandwidth,
                             sum_weights,
                             kernel,
                             metric,
                             metric_arg);
}

}  // namespace ML::KDE
