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

#include <distance/distance.h>
#include <distance/fused_l2_nn.h>
#include <linalg/binary_op.h>
#include <linalg/matrix_vector_op.h>
#include <linalg/mean_squared_error.h>
#include <linalg/reduce.h>
#include <linalg/reduce_rows_by_key.h>
#include <matrix/gather.h>
#include <random/permute.h>
#include <random/rng.h>
#include <random>

#include <ml_cuda_utils.h>

#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/scan.h>
#include <numeric>

#include <common/allocatorAdapter.hpp>
#include <common/cumlHandle.hpp>
#include <common/cuml_comms_int.hpp>
#include <common/device_buffer.hpp>
#include <common/host_buffer.hpp>
#include <common/tensor.hpp>

#include <cuml/cluster/kmeans.hpp>

#include <fstream>

namespace ML {

//@todo: Use GLOG once https://github.com/rapidsai/cuml/issues/100 is addressed.
#define LOG(handle, verbose, fmt, ...)                                   \
  do {                                                                   \
    bool verbose_ = verbose;                                             \
    if (handle.commsInitialized()) {                                     \
      const MLCommon::cumlCommunicator &comm = handle.getCommunicator(); \
      const int my_rank = comm.getRank();                                \
      verbose_ = verbose && (my_rank == 0);                              \
    }                                                                    \
    if (verbose_) {                                                      \
      std::string msg;                                                   \
      char verboseMsg[2048];                                             \
      std::sprintf(verboseMsg, fmt, ##__VA_ARGS__);                      \
      msg += verboseMsg;                                                 \
      std::cerr << msg;                                                  \
    }                                                                    \
  } while (0)

namespace kmeans {
namespace detail {

template <typename LabelT, typename DataT>
struct FusedL2NNReduceOp {
  LabelT offset;

  FusedL2NNReduceOp(LabelT _offset) : offset(_offset){};

  typedef typename cub::KeyValuePair<LabelT, DataT> KVP;
  DI void operator()(KVP *out, const KVP &other) {
    if (other.value < out->value) {
      out->key = offset + other.key;
      out->value = other.value;
    }
  }

  DI void operator()(DataT *out, const KVP &other) {
    if (other.value < *out) {
      *out = other.value;
    }
  }

  DI void init(DataT *out, DataT maxVal) { *out = maxVal; }
  DI void init(KVP *out, DataT maxVal) {
    out->key = -1;
    out->value = maxVal;
  }
};

template <typename DataT>
struct SamplingOp {
  DataT *rnd;
  int *flag;
  DataT cluster_cost;
  double oversampling_factor;
  int n_clusters;

  CUB_RUNTIME_FUNCTION __forceinline__ SamplingOp(DataT c, double l, int k,
                                                  DataT *rand, int *ptr)
    : cluster_cost(c),
      oversampling_factor(l),
      n_clusters(k),
      rnd(rand),
      flag(ptr) {}

  __host__ __device__ __forceinline__ bool operator()(
    const cub::KeyValuePair<ptrdiff_t, DataT> &a) const {
    DataT prob_threshold = (DataT)rnd[a.key];

    DataT prob_x =
      ((oversampling_factor * n_clusters * a.value) / cluster_cost);

    return !flag[a.key] && (prob_x > prob_threshold);
  }
};

template <typename IndexT, typename DataT>
struct KeyValueIndexOp {
  __host__ __device__ __forceinline__ IndexT
  operator()(const cub::KeyValuePair<IndexT, DataT> &a) const {
    return a.key;
  }
};

template <typename CountT>
CountT getDataBatchSize(const KMeansParams &params, CountT n_samples) {
  auto minVal = std::min(params.batch_samples, n_samples);
  return (minVal == 0) ? n_samples : minVal;
}

template <typename CountT>
CountT getCentroidsBatchSize(const KMeansParams &params,
                             CountT n_local_clusters) {
  auto minVal = std::min(params.batch_centroids, n_local_clusters);
  return (minVal == 0) ? n_local_clusters : minVal;
}

// Computes the intensity histogram from a sequence of labels
template <typename SampleIteratorT, typename CounterT>
void countLabels(const cumlHandle_impl &handle, SampleIteratorT labels,
                 CounterT *count, int n_samples, int n_clusters,
                 MLCommon::device_buffer<char> &workspace,
                 cudaStream_t stream) {
  int num_levels = n_clusters + 1;
  int lower_level = 0;
  int upper_level = n_clusters;

  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceHistogram::HistogramEven(
    nullptr, temp_storage_bytes, labels, count, num_levels, lower_level,
    upper_level, n_samples, stream));

  workspace.resize(temp_storage_bytes, stream);

  CUDA_CHECK(cub::DeviceHistogram::HistogramEven(
    workspace.data(), temp_storage_bytes, labels, count, num_levels,
    lower_level, upper_level, n_samples, stream));
}

template <typename DataT, typename IndexT>
Tensor<DataT, 2, IndexT> sampleCentroids(
  const cumlHandle_impl &handle, Tensor<DataT, 2, IndexT> &X,
  Tensor<DataT, 1, IndexT> &minClusterDistance,
  Tensor<int, 1, IndexT> &isSampleCentroid,
  typename kmeans::detail::SamplingOp<DataT> &select_op,
  MLCommon::device_buffer<char> &workspace, cudaStream_t stream) {
  int n_local_samples = X.getSize(0);
  int n_features = X.getSize(1);

  Tensor<int, 1> nSelected({1}, handle.getDeviceAllocator(), stream);

  cub::ArgIndexInputIterator<DataT *> ip_itr(minClusterDistance.data());
  Tensor<cub::KeyValuePair<ptrdiff_t, DataT>, 1> sampledMinClusterDistance(
    {n_local_samples}, handle.getDeviceAllocator(), stream);
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceSelect::If(
    nullptr, temp_storage_bytes, ip_itr, sampledMinClusterDistance.data(),
    nSelected.data(), n_local_samples, select_op, stream));

  workspace.resize(temp_storage_bytes, stream);

  CUDA_CHECK(cub::DeviceSelect::If(workspace.data(), temp_storage_bytes, ip_itr,
                                   sampledMinClusterDistance.data(),
                                   nSelected.data(), n_local_samples, select_op,
                                   stream));

  int nPtsSampledInRank = 0;
  MLCommon::copy(&nPtsSampledInRank, nSelected.data(), nSelected.numElements(),
                 stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  int *rawPtr_isSampleCentroid = isSampleCentroid.data();
  ML::thrustAllocatorAdapter alloc(handle.getDeviceAllocator(), stream);
  auto execution_policy = thrust::cuda::par(alloc).on(stream);
  thrust::for_each_n(execution_policy, sampledMinClusterDistance.begin(),
                     nPtsSampledInRank,
                     [=] __device__(cub::KeyValuePair<ptrdiff_t, DataT> val) {
                       rawPtr_isSampleCentroid[val.key] = 1;
                     });

  Tensor<DataT, 2, IndexT> inRankCp({nPtsSampledInRank, n_features},
                                    handle.getDeviceAllocator(), stream);

  MLCommon::Matrix::gather(
    X.data(), X.getSize(1), X.getSize(0), sampledMinClusterDistance.data(),
    nPtsSampledInRank, inRankCp.data(),
    [=] __device__(cub::KeyValuePair<ptrdiff_t, DataT> val) {  // MapTransformOp
      return val.key;
    },
    stream);

  return inRankCp;
}

template <typename DataT, typename IndexT, typename ReductionOpT>
void computeClusterCost(const cumlHandle_impl &handle,
                        Tensor<DataT, 1, IndexT> &minClusterDistance,
                        MLCommon::device_buffer<char> &workspace,
                        DataT *clusterCost, ReductionOpT reduction_op,
                        cudaStream_t stream) {
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceReduce::Reduce(
    nullptr, temp_storage_bytes, minClusterDistance.data(), clusterCost,
    minClusterDistance.numElements(), reduction_op, DataT(), stream));

  workspace.resize(temp_storage_bytes, stream);

  CUDA_CHECK(cub::DeviceReduce::Reduce(workspace.data(), temp_storage_bytes,
                                       minClusterDistance.data(), clusterCost,
                                       minClusterDistance.numElements(),
                                       reduction_op, DataT(), stream));
}

// calculate pairwise distance between 'dataset[n x d]' and 'centroids[k x d]',
// result will be stored in 'pairwiseDistance[n x k]'
template <typename DataT, typename IndexT>
void pairwiseDistance(const cumlHandle_impl &handle,
                      Tensor<DataT, 2, IndexT> &X,
                      Tensor<DataT, 2, IndexT> &centroids,
                      Tensor<DataT, 2, IndexT> &pairwiseDistance,
                      MLCommon::device_buffer<char> &workspace,
                      MLCommon::Distance::DistanceType metric,
                      cudaStream_t stream) {
  auto n_samples = X.getSize(0);
  auto n_features = X.getSize(1);
  auto n_clusters = centroids.getSize(0);

  ASSERT(X.getSize(1) == centroids.getSize(1),
         "# features in dataset and centroids are different (must be same)");
  MLCommon::Distance::pairwiseDistance<DataT, IndexT>(
    X.data(), centroids.data(), pairwiseDistance.data(), n_samples, n_clusters,
    n_features, workspace, metric, stream);
}

// Calculates a <key, value> pair for every sample in input 'X' where key is an
// index to an sample in 'centroids' (index of the nearest centroid) and 'value'
// is the distance between the sample and the 'centroid[key]'
template <typename DataT, typename IndexT>
void minClusterAndDistance(
  const cumlHandle_impl &handle, const KMeansParams &params,
  Tensor<DataT, 2, IndexT> &X, Tensor<DataT, 2, IndexT> &centroids,
  Tensor<cub::KeyValuePair<IndexT, DataT>, 1, IndexT> &minClusterAndDistance,
  Tensor<DataT, 1, IndexT> &L2NormX,
  MLCommon::device_buffer<DataT> &L2NormBuf_OR_DistBuf,
  MLCommon::device_buffer<char> &workspace,
  MLCommon::Distance::DistanceType metric, cudaStream_t stream) {
  auto n_samples = X.getSize(0);
  auto n_features = X.getSize(1);
  auto n_clusters = centroids.getSize(0);
  auto dataBatchSize = kmeans::detail::getDataBatchSize(params, n_samples);
  auto centroidsBatchSize =
    kmeans::detail::getCentroidsBatchSize(params, n_clusters);

  if (metric == MLCommon::Distance::EucExpandedL2 ||
      metric == MLCommon::Distance::EucExpandedL2Sqrt) {
    L2NormBuf_OR_DistBuf.resize(n_clusters, stream);
    MLCommon::LinAlg::rowNorm(L2NormBuf_OR_DistBuf.data(), centroids.data(),
                              centroids.getSize(1), centroids.getSize(0),
                              MLCommon::LinAlg::L2Norm, true, stream);
  } else {
    L2NormBuf_OR_DistBuf.resize(dataBatchSize * centroidsBatchSize, stream);
  }

  // Note - pairwiseDistance and centroidsNorm share the same buffer
  // centroidsNorm [n_clusters] - tensor wrapper around centroids L2 Norm
  Tensor<DataT, 1> centroidsNorm(L2NormBuf_OR_DistBuf.data(), {n_clusters});
  // pairwiseDistance[ns x nc] - tensor wrapper around the distance buffer
  Tensor<DataT, 2, IndexT> pairwiseDistance(
    L2NormBuf_OR_DistBuf.data(), {dataBatchSize, centroidsBatchSize});

  cub::KeyValuePair<IndexT, DataT> initial_value(
    0, std::numeric_limits<DataT>::max());

  ML::thrustAllocatorAdapter alloc(handle.getDeviceAllocator(), stream);
  auto thrust_exec_policy = thrust::cuda::par(alloc).on(stream);
  thrust::fill(thrust_exec_policy, minClusterAndDistance.begin(),
               minClusterAndDistance.end(), initial_value);

  // tile over the input dataset
  for (auto dIdx = 0; dIdx < n_samples; dIdx += dataBatchSize) {
    // # of samples for the current batch
    auto ns = std::min(dataBatchSize, n_samples - dIdx);

    // datasetView [ns x n_features] - view representing the current batch of
    // input dataset
    auto datasetView = X.template view<2>({ns, n_features}, {dIdx, 0});

    // minClusterAndDistanceView [ns x n_clusters]
    auto minClusterAndDistanceView =
      minClusterAndDistance.template view<1>({ns}, {dIdx});

    auto L2NormXView = L2NormX.template view<1>({ns}, {dIdx});

    // tile over the centroids
    for (auto cIdx = 0; cIdx < n_clusters; cIdx += centroidsBatchSize) {
      // # of centroids for the current batch
      auto nc = std::min(centroidsBatchSize, n_clusters - cIdx);

      // centroidsView [nc x n_features] - view representing the current batch
      // of centroids
      auto centroidsView =
        centroids.template view<2>({nc, n_features}, {cIdx, 0});

      if (metric == MLCommon::Distance::EucExpandedL2 ||
          metric == MLCommon::Distance::EucExpandedL2Sqrt) {
        auto centroidsNormView = centroidsNorm.template view<1>({nc}, {cIdx});
        workspace.resize((sizeof(int)) * ns, stream);

        FusedL2NNReduceOp<IndexT, DataT> redOp(cIdx);

        MLCommon::Distance::fusedL2NN<DataT, cub::KeyValuePair<IndexT, DataT>,
                                      IndexT>(
          minClusterAndDistanceView.data(), datasetView.data(),
          centroidsView.data(), L2NormXView.data(), centroidsNormView.data(),
          ns, nc, n_features, (void *)workspace.data(), redOp,
          (metric == MLCommon::Distance::EucExpandedL2) ? false : true, false,
          stream);
      } else {
        // pairwiseDistanceView [ns x nc] - view representing the pairwise
        // distance for current batch
        auto pairwiseDistanceView =
          pairwiseDistance.template view<2>({ns, nc}, {0, 0});

        // calculate pairwise distance between current tile of cluster centroids
        // and input dataset
        kmeans::detail::pairwiseDistance(handle, datasetView, centroidsView,
                                         pairwiseDistanceView, workspace,
                                         metric, stream);

        // argmin reduction returning <index, value> pair
        // calculates the closest centroid and the distance to the closest
        // centroid
        MLCommon::LinAlg::coalescedReduction(
          minClusterAndDistanceView.data(), pairwiseDistanceView.data(),
          pairwiseDistanceView.getSize(1), pairwiseDistanceView.getSize(0),
          initial_value, stream, true,
          [=] __device__(const DataT val, const IndexT i) {
            cub::KeyValuePair<IndexT, DataT> pair;
            pair.key = cIdx + i;
            pair.value = val;
            return pair;
          },
          [=] __device__(cub::KeyValuePair<IndexT, DataT> a,
                         cub::KeyValuePair<IndexT, DataT> b) {
            return (b.value < a.value) ? b : a;
          },
          [=] __device__(cub::KeyValuePair<IndexT, DataT> pair) {
            return pair;
          });
      }
    }
  }
}

template <typename DataT, typename IndexT>
void minClusterDistance(const cumlHandle_impl &handle,
                        const KMeansParams &params, Tensor<DataT, 2, IndexT> &X,
                        Tensor<DataT, 2, IndexT> &centroids,
                        Tensor<DataT, 1, IndexT> &minClusterDistance,
                        Tensor<DataT, 1, IndexT> &L2NormX,
                        MLCommon::device_buffer<DataT> &L2NormBuf_OR_DistBuf,
                        MLCommon::device_buffer<char> &workspace,
                        MLCommon::Distance::DistanceType metric,
                        cudaStream_t stream) {
  auto n_samples = X.getSize(0);
  auto n_features = X.getSize(1);
  auto n_clusters = centroids.getSize(0);

  auto dataBatchSize = kmeans::detail::getDataBatchSize(params, n_samples);
  auto centroidsBatchSize =
    kmeans::detail::getCentroidsBatchSize(params, n_clusters);

  if (metric == MLCommon::Distance::EucExpandedL2 ||
      metric == MLCommon::Distance::EucExpandedL2Sqrt) {
    L2NormBuf_OR_DistBuf.resize(n_clusters, stream);
    MLCommon::LinAlg::rowNorm(L2NormBuf_OR_DistBuf.data(), centroids.data(),
                              centroids.getSize(1), centroids.getSize(0),
                              MLCommon::LinAlg::L2Norm, true, stream);
  } else {
    L2NormBuf_OR_DistBuf.resize(dataBatchSize * centroidsBatchSize, stream);
  }

  // Note - pairwiseDistance and centroidsNorm share the same buffer
  // centroidsNorm [n_clusters] - tensor wrapper around centroids L2 Norm
  Tensor<DataT, 1> centroidsNorm(L2NormBuf_OR_DistBuf.data(), {n_clusters});
  // pairwiseDistance[ns x nc] - tensor wrapper around the distance buffer
  Tensor<DataT, 2, IndexT> pairwiseDistance(
    L2NormBuf_OR_DistBuf.data(), {dataBatchSize, centroidsBatchSize});

  ML::thrustAllocatorAdapter alloc(handle.getDeviceAllocator(), stream);
  auto thrust_exec_policy = thrust::cuda::par(alloc).on(stream);
  thrust::fill(thrust_exec_policy, minClusterDistance.begin(),
               minClusterDistance.end(), std::numeric_limits<DataT>::max());

  // tile over the input data and calculate distance matrix [n_samples x
  // n_clusters]
  for (int dIdx = 0; dIdx < n_samples; dIdx += dataBatchSize) {
    // # of samples for the current batch
    auto ns = std::min(dataBatchSize, n_samples - dIdx);

    // datasetView [ns x n_features] - view representing the current batch of
    // input dataset
    auto datasetView = X.template view<2>({ns, n_features}, {dIdx, 0});

    // minClusterDistanceView [ns x n_clusters]
    auto minClusterDistanceView =
      minClusterDistance.template view<1>({ns}, {dIdx});

    auto L2NormXView = L2NormX.template view<1>({ns}, {dIdx});

    // tile over the centroids
    for (auto cIdx = 0; cIdx < n_clusters; cIdx += centroidsBatchSize) {
      // # of centroids for the current batch
      auto nc = std::min(centroidsBatchSize, n_clusters - cIdx);

      // centroidsView [nc x n_features] - view representing the current batch
      // of centroids
      auto centroidsView =
        centroids.template view<2>({nc, n_features}, {cIdx, 0});

      if (metric == MLCommon::Distance::EucExpandedL2 ||
          metric == MLCommon::Distance::EucExpandedL2Sqrt) {
        auto centroidsNormView = centroidsNorm.template view<1>({nc}, {cIdx});
        workspace.resize((sizeof(int)) * ns, stream);

        FusedL2NNReduceOp<IndexT, DataT> redOp(cIdx);
        MLCommon::Distance::fusedL2NN<DataT, DataT, IndexT>(
          minClusterDistanceView.data(), datasetView.data(),
          centroidsView.data(), L2NormXView.data(), centroidsNormView.data(),
          ns, nc, n_features, (void *)workspace.data(), redOp,
          (metric == MLCommon::Distance::EucExpandedL2) ? false : true, false,
          stream);
      } else {
        // pairwiseDistanceView [ns x nc] - view representing the pairwise
        // distance for current batch
        auto pairwiseDistanceView =
          pairwiseDistance.template view<2>({ns, nc}, {0, 0});

        // calculate pairwise distance between current tile of cluster centroids
        // and input dataset
        kmeans::detail::pairwiseDistance(handle, datasetView, centroidsView,
                                         pairwiseDistanceView, workspace,
                                         metric, stream);

        MLCommon::LinAlg::coalescedReduction(
          minClusterDistanceView.data(), pairwiseDistanceView.data(),
          pairwiseDistanceView.getSize(1), pairwiseDistanceView.getSize(0),
          std::numeric_limits<DataT>::max(), stream, true,
          [=] __device__(DataT val, int i) {  // MainLambda
            return val;
          },
          [=] __device__(DataT a, DataT b) {  // ReduceLambda
            return (b < a) ? b : a;
          },
          [=] __device__(DataT val) {  // FinalLambda
            return val;
          });
      }
    }
  }
}

// shuffle and randomly select 'n_samples_to_gather' from input 'in' and stores
// in 'out' does not modify the input
template <typename DataT, typename IndexT>
void shuffleAndGather(const cumlHandle_impl &handle,
                      const Tensor<DataT, 2, IndexT> &in,
                      Tensor<DataT, 2, IndexT> &out, size_t n_samples_to_gather,
                      int seed, cudaStream_t stream,
                      MLCommon::device_buffer<char> *workspace = nullptr) {
  auto n_samples = in.getSize(0);
  auto n_features = in.getSize(1);

  Tensor<IndexT, 1> indices({n_samples}, handle.getDeviceAllocator(), stream);

  if (workspace) {
    // shuffle indices on device using ml-prims
    MLCommon::Random::permute<DataT>(indices.data(), nullptr, nullptr,
                                     in.getSize(1), in.getSize(0), true,
                                     stream);
  } else {
    // shuffle indices on host and copy to device...
    MLCommon::host_buffer<IndexT> ht_indices(handle.getHostAllocator(), stream,
                                             n_samples);

    std::iota(ht_indices.begin(), ht_indices.end(), 0);

    std::mt19937 gen(seed);
    std::shuffle(ht_indices.begin(), ht_indices.end(), gen);

    MLCommon::copy(indices.data(), ht_indices.data(), indices.numElements(),
                   stream);
  }

  MLCommon::Matrix::gather(in.data(), in.getSize(1), in.getSize(0),
                           indices.data(), n_samples_to_gather, out.data(),
                           stream);
}

template <typename DataT, typename IndexT>
void countSamplesInCluster(
  const cumlHandle_impl &handle, const KMeansParams &params,
  Tensor<DataT, 2, IndexT> &X, Tensor<DataT, 1, IndexT> &L2NormX,
  Tensor<DataT, 2, IndexT> &centroids, MLCommon::device_buffer<char> &workspace,
  MLCommon::Distance::DistanceType metric,
  Tensor<int, 1, IndexT> &sampleCountInCluster, cudaStream_t stream) {
  auto n_samples = X.getSize(0);
  auto n_features = X.getSize(1);
  auto n_clusters = centroids.getSize(0);

  // stores (key, value) pair corresponding to each sample where
  //   - key is the index of nearest cluster
  //   - value is the distance to the nearest cluster
  Tensor<cub::KeyValuePair<IndexT, DataT>, 1, IndexT> minClusterAndDistance(
    {n_samples}, handle.getDeviceAllocator(), stream);

  // temporary buffer to store distance matrix, destructor releases the resource
  MLCommon::device_buffer<DataT> L2NormBuf_OR_DistBuf(
    handle.getDeviceAllocator(), stream);

  // computes minClusterAndDistance[0:n_samples) where  minClusterAndDistance[i]
  // is a <key, value> pair where
  //   'key' is index to an sample in 'centroids' (index of the nearest
  //   centroid) and 'value' is the distance between the sample 'X[i]' and the
  //   'centroid[key]'
  kmeans::detail::minClusterAndDistance(
    handle, params, X, centroids, minClusterAndDistance, L2NormX,
    L2NormBuf_OR_DistBuf, workspace, metric, stream);

  // Using TransformInputIteratorT to dereference an array of cub::KeyValuePair
  // and converting them to just return the Key to be used in reduce_rows_by_key
  // prims
  kmeans::detail::KeyValueIndexOp<IndexT, DataT> conversion_op;
  cub::TransformInputIterator<IndexT,
                              kmeans::detail::KeyValueIndexOp<IndexT, DataT>,
                              cub::KeyValuePair<IndexT, DataT> *>
    itr(minClusterAndDistance.data(), conversion_op);

  // count # of samples in each cluster
  kmeans::detail::countLabels(handle, itr, sampleCountInCluster.data(),
                              n_samples, n_clusters, workspace, stream);
}

/*
 * @brief Selects 'n_clusters' samples from the input X using kmeans++ algorithm.

 * @note  This is the algorithm described in
 *        "k-means++: the advantages of careful seeding". 2007, Arthur, D. and Vassilvitskii, S.
 *        ACM-SIAM symposium on Discrete algorithms.
 *
 * Scalable kmeans++ pseudocode
 * 1: C = sample a point uniformly at random from X
 * 2: while |C| < k
 * 3:   Sample x in X with probability p_x = d^2(x, C) / phi_X (C) 
 * 4:   C = C U {x}
 * 5: end for
 */
template <typename DataT, typename IndexT>
void kmeansPlusPlus(const cumlHandle_impl &handle, const KMeansParams &params,
                    Tensor<DataT, 2, IndexT> &X,
                    MLCommon::Distance::DistanceType metric,
                    MLCommon::device_buffer<char> &workspace,
                    MLCommon::device_buffer<DataT> &centroidsRawData,
                    cudaStream_t stream) {
  auto n_samples = X.getSize(0);
  auto n_features = X.getSize(1);
  auto n_clusters = params.n_clusters;

  // number of seeding trials for each center (except the first)
  auto n_trials = 2 + static_cast<int>(std::ceil(log(n_clusters)));

  LOG(handle, params.verbose,
      "Run sequential k-means++ to select %d centroids from %d input samples "
      "(%d seeding trials per iterations)\n",
      n_clusters, n_samples, n_trials);

  auto dataBatchSize = kmeans::detail::getDataBatchSize(params, n_samples);

  // temporary buffers
  MLCommon::host_buffer<DataT> h_wt(handle.getHostAllocator(), stream,
                                    n_samples);

  MLCommon::device_buffer<DataT> distBuffer(handle.getDeviceAllocator(), stream,
                                            n_trials * n_samples);

  Tensor<DataT, 2, IndexT> centroidCandidates(
    {n_trials, n_features}, handle.getDeviceAllocator(), stream);

  Tensor<DataT, 1, IndexT> costPerCandidate(
    {n_trials}, handle.getDeviceAllocator(), stream);

  Tensor<DataT, 1, IndexT> minClusterDistance(
    {n_samples}, handle.getDeviceAllocator(), stream);

  MLCommon::device_buffer<DataT> L2NormBuf_OR_DistBuf(
    handle.getDeviceAllocator(), stream);

  MLCommon::device_buffer<DataT> clusterCost(handle.getDeviceAllocator(),
                                             stream, 1);

  MLCommon::device_buffer<cub::KeyValuePair<int, DataT>>
    minClusterIndexAndDistance(handle.getDeviceAllocator(), stream, 1);

  // L2 norm of X: ||c||^2
  Tensor<DataT, 1> L2NormX({n_samples}, handle.getDeviceAllocator(), stream);

  if (metric == MLCommon::Distance::EucExpandedL2 ||
      metric == MLCommon::Distance::EucExpandedL2Sqrt) {
    MLCommon::LinAlg::rowNorm(L2NormX.data(), X.data(), X.getSize(1),
                              X.getSize(0), MLCommon::LinAlg::L2Norm, true,
                              stream);
  }

  std::mt19937 gen(params.seed);
  std::uniform_int_distribution<> dis(0, n_samples - 1);

  ML::thrustAllocatorAdapter alloc(handle.getDeviceAllocator(), stream);
  auto thrust_exec_policy = thrust::cuda::par(alloc).on(stream);

  // <<< Step-1 >>>: C <-- sample a point uniformly at random from X
  auto initialCentroid = X.template view<2>({1, n_features}, {dis(gen), 0});
  int n_clusters_picked = 1;

  // reset buffer to store the chosen centroid
  centroidsRawData.reserve(n_clusters * n_features, stream);
  centroidsRawData.resize(initialCentroid.numElements(), stream);
  MLCommon::copy(centroidsRawData.begin(), initialCentroid.data(),
                 initialCentroid.numElements(), stream);

  //  C = initial set of centroids
  auto centroids = std::move(Tensor<DataT, 2, IndexT>(
    centroidsRawData.data(),
    {initialCentroid.getSize(0), initialCentroid.getSize(1)}));
  // <<< End of Step-1 >>>

  // Calculate cluster distance, d^2(x, C), for all the points x in X to the nearest centroid
  kmeans::detail::minClusterDistance(
    handle, params, X, centroids, minClusterDistance, L2NormX,
    L2NormBuf_OR_DistBuf, workspace, metric, stream);

  LOG(handle, params.verbose, " k-means++ - Sampled %d/%d centroids\n",
      n_clusters_picked, n_clusters);

  // <<<< Step-2 >>> : while |C| < k
  while (n_clusters_picked < n_clusters) {
    // <<< Step-3 >>> : Sample x in X with probability p_x = d^2(x, C) / phi_X (C)
    // Choose 'n_trials' centroid candidates from X with probability proportional to the squared distance to the nearest existing cluster
    MLCommon::copy(h_wt.data(), minClusterDistance.data(),
                   minClusterDistance.numElements(), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Note - n_trials is relative small here, we don't need MLCommon::gather call
    std::discrete_distribution<> d(h_wt.begin(), h_wt.end());
    for (int cIdx = 0; cIdx < n_trials; ++cIdx) {
      auto rand_idx = d(gen);
      auto randCentroid = X.template view<2>({1, n_features}, {rand_idx, 0});
      MLCommon::copy(centroidCandidates.data() + cIdx * n_features,
                     randCentroid.data(), randCentroid.numElements(), stream);
    }

    // Calculate pairwise distance between X and the centroid candidates
    // Output - pwd [n_trails x n_samples]
    auto pwd = std::move(
      Tensor<DataT, 2, IndexT>(distBuffer.data(), {n_trials, n_samples}));
    kmeans::detail::pairwiseDistance(handle, centroidCandidates, X, pwd,
                                     workspace, metric, stream);

    // Update nearest cluster distance for each centroid candidate
    // Note pwd and minDistBuf points to same buffer which currently holds pairwise distance values.
    // Outputs minDistanceBuf[m_trails x n_samples] where minDistance[i, :] contains updated minClusterDistance that includes candidate-i
    auto minDistBuf = std::move(
      Tensor<DataT, 2, IndexT>(distBuffer.data(), {n_trials, n_samples}));
    MLCommon::LinAlg::matrixVectorOp(
      minDistBuf.data(), pwd.data(), minClusterDistance.data(), pwd.getSize(1),
      pwd.getSize(0), true, true,
      [=] __device__(DataT mat, DataT vec) { return vec <= mat ? vec : mat; },
      stream);

    // Calculate costPerCandidate[n_trials] where costPerCandidate[i] is the cluster cost when using centroid candidate-i
    MLCommon::LinAlg::reduce(costPerCandidate.data(), minDistBuf.data(),
                             minDistBuf.getSize(1), minDistBuf.getSize(0),
                             static_cast<DataT>(0), true, true, stream);

    // Greedy Choice - Choose the candidate that has minimum cluster cost
    // ArgMin operation below identifies the index of minimum cost in costPerCandidate
    {
      // Determine temporary device storage requirements
      size_t temp_storage_bytes = 0;
      cub::DeviceReduce::ArgMin(
        nullptr, temp_storage_bytes, costPerCandidate.data(),
        minClusterIndexAndDistance.data(), costPerCandidate.getSize(0));

      // Allocate temporary storage
      workspace.resize(temp_storage_bytes, stream);

      // Run argmin-reduction
      cub::DeviceReduce::ArgMin(
        workspace.data(), temp_storage_bytes, costPerCandidate.data(),
        minClusterIndexAndDistance.data(), costPerCandidate.getSize(0));

      int bestCandidateIdx = -1;
      MLCommon::copy(&bestCandidateIdx, &minClusterIndexAndDistance.data()->key,
                     1, stream);
      /// <<< End of Step-3 >>>

      /// <<< Step-4 >>>: C = C U {x}
      // Update minimum cluster distance corresponding to the chosen centroid candidate
      MLCommon::copy(minClusterDistance.data(),
                     minDistBuf.data() + bestCandidateIdx * n_samples,
                     n_samples, stream);

      MLCommon::copy(centroidsRawData.data() + n_clusters_picked * n_features,
                     centroidCandidates.data() + bestCandidateIdx * n_features,
                     n_features, stream);

      ++n_clusters_picked;
      /// <<< End of Step-4 >>>
    }

    LOG(handle, params.verbose, " k-means++ - Sampled %d/%d centroids\n",
        n_clusters_picked, n_clusters);
  }  /// <<<< Step-5 >>>
}

};  // namespace detail
};  // namespace kmeans
};  // namespace ML
