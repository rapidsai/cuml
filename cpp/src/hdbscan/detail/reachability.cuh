/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include "kernels/reachability.cuh"

#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>

#include <raft/mr/device/buffer.hpp>

#include <raft/linalg/unary_op.cuh>

#include <faiss/impl/AuxIndexStructures.h>
#include <cuml/common/logger.hpp>
#include <faiss/gpu/impl/Distance.cuh>
#include <faiss/gpu/impl/DistanceUtils.cuh>
#include <faiss/gpu/impl/L2Norm.cuh>
#include <faiss/gpu/impl/L2Select.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/MatrixMult.cuh>

#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/hierarchy/detail/connectivities.cuh>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/sparse/selection/knn_graph.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuml/neighbors/knn.hpp>
#include <raft/distance/distance.cuh>

#include <thrust/transform.h>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Reachability {

/**
 * Computes expanded L2 metric, projects points into reachability
 * space, and performs a k-select.
 * @tparam value_t
 * @param[in] productDistances Tensor (or blocked view) of inner products
 * @param[in] centroidDistances Tensor of l2 norms
 * @param[in] coreDistances Tensor of core distances
 * @param[out] outDistances Tensor of output distances
 * @param[out] outIndices Tensor of output indices
 * @param[in] batch_offset starting row (used when productDistances is a batch)
 * @param[in] k number of neighbors to select
 * @param[in] stream cuda stream for ordering gpu computations
 */
template <typename value_t>
void runL2SelectMin(faiss::gpu::Tensor<value_t, 2, true> &productDistances,
                    faiss::gpu::Tensor<value_t, 1, true> &centroidDistances,
                    faiss::gpu::Tensor<value_t, 1, true> &coreDistances,
                    faiss::gpu::Tensor<value_t, 2, true> &outDistances,
                    faiss::gpu::Tensor<int, 2, true> &outIndices,
                    int batch_offset, int k, value_t alpha,
                    cudaStream_t stream) {
  FAISS_ASSERT(productDistances.getSize(0) == outDistances.getSize(0));
  FAISS_ASSERT(productDistances.getSize(0) == outIndices.getSize(0));
  //  FAISS_ASSERT(centroidDistances.getSize(0) == productDistances.getSize(1));
  FAISS_ASSERT(outDistances.getSize(1) == k);
  FAISS_ASSERT(outIndices.getSize(1) == k);
  FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);

  auto grid = dim3(outDistances.getSize(0));

#define RUN_L2_SELECT(BLOCK, NUM_WARP_Q, NUM_THREAD_Q)                      \
  do {                                                                      \
    l2SelectMinK<value_t, NUM_WARP_Q, NUM_THREAD_Q, BLOCK>                  \
      <<<grid, BLOCK, 0, stream>>>(                                         \
        productDistances, centroidDistances, coreDistances, outDistances,   \
        outIndices, batch_offset, k, faiss::gpu::Limits<value_t>::getMax(), \
        alpha);                                                             \
  } while (0)

  // block size 128 for everything <= 1024
  if (k <= 32) {
    RUN_L2_SELECT(128, 32, 2);
  } else if (k <= 64) {
    RUN_L2_SELECT(128, 64, 3);
  } else if (k <= 128) {
    RUN_L2_SELECT(128, 128, 3);
  } else if (k <= 256) {
    RUN_L2_SELECT(128, 256, 4);
  } else if (k <= 512) {
    RUN_L2_SELECT(128, 512, 8);
  } else if (k <= 1024) {
    RUN_L2_SELECT(128, 1024, 8);

#if GPU_MAX_SELECTION_K >= 2048
  } else if (k <= 2048) {
    // smaller block for less shared memory
    RUN_L2_SELECT(64, 2048, 8);
#endif

  } else {
    FAISS_ASSERT(false);
  }
}

/**
 * Given core distances, Fuses computations of L2 distances between all
 * points, projection into mutual reachability space, and k-selection.
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource reuse
 * @param[out] out_inds  output indices array (size m * k)
 * @param[out] out_dists output distances array (size m * k)
 * @param[in] X input data points (size m * n)
 * @param[in] m number of rows in X
 * @param[in] n number of columns in X
 * @param[in] k neighborhood size (includes self-loop)
 * @param[in] core_dists array of core distances (size m)
 */
template <typename value_idx, typename value_t>
void mutual_reachability_knn_l2(const raft::handle_t &handle,
                                value_idx *out_inds, value_t *out_dists,
                                const value_t *X, size_t m, size_t n, int k,
                                value_t *core_dists, value_t alpha) {
  auto device = faiss::gpu::getCurrentDevice();
  auto stream = handle.get_stream();

  faiss::gpu::DeviceScope ds(device);
  faiss::gpu::StandardGpuResources res;

  res.noTempMemory();
  res.setDefaultStream(device, stream);

  auto resImpl = res.getResources();
  auto gpu_res = resImpl.get();

  gpu_res->initializeForDevice(device);
  gpu_res->setDefaultStream(device, stream);

  device = faiss::gpu::getCurrentDevice();

  auto tmp_mem_cur_device = gpu_res->getTempMemoryAvailableCurrentDevice();

  /**
   * Compute L2 norm
   */
  rmm::device_uvector<value_t> norms(m, stream);

  auto core_dists_tensor = faiss::gpu::toDeviceTemporary<value_t, 1>(
    gpu_res, device,
    const_cast<value_t *>(reinterpret_cast<const value_t *>(core_dists)),
    stream, {(int)m});

  auto x_tensor = faiss::gpu::toDeviceTemporary<value_t, 2>(
    gpu_res, device,
    const_cast<value_t *>(reinterpret_cast<const value_t *>(X)), stream,
    {(int)m, (int)n});

  auto out_dists_tensor = faiss::gpu::toDeviceTemporary<value_t, 2>(
    gpu_res, device,
    const_cast<value_t *>(reinterpret_cast<const value_t *>(out_dists)), stream,
    {(int)m, k});

  auto out_inds_tensor = faiss::gpu::toDeviceTemporary<value_idx, 2>(
    gpu_res, device,
    const_cast<value_idx *>(reinterpret_cast<const value_idx *>(out_inds)),
    stream, {(int)m, k});

  auto norms_tensor = faiss::gpu::toDeviceTemporary<value_t, 1>(
    gpu_res, device,
    const_cast<value_t *>(reinterpret_cast<const value_t *>(norms.data())),
    stream, {(int)m});

  runL2Norm(x_tensor, true, norms_tensor, true, stream);

  /**
   * Tile over PW dists, accumulating k-select
   */

  int tileRows = 0;
  int tileCols = 0;
  faiss::gpu::chooseTileSize(m, m, n, sizeof(value_t), tmp_mem_cur_device,
                             tileRows, tileCols);

  int numColTiles = raft::ceildiv(m, (size_t)tileCols);

  faiss::gpu::DeviceTensor<value_t, 2, true> distanceBuf1(
    gpu_res, faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream),
    {tileRows, tileCols});
  faiss::gpu::DeviceTensor<value_t, 2, true> distanceBuf2(
    gpu_res, faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream),
    {tileRows, tileCols});

  faiss::gpu::DeviceTensor<value_t, 2, true> *distanceBufs[2] = {&distanceBuf1,
                                                                 &distanceBuf2};

  faiss::gpu::DeviceTensor<value_t, 2, true> outDistanceBuf1(
    gpu_res, faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream),
    {tileRows, numColTiles * k});
  faiss::gpu::DeviceTensor<value_t, 2, true> outDistanceBuf2(
    gpu_res, faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream),
    {tileRows, numColTiles * k});
  faiss::gpu::DeviceTensor<value_t, 2, true> *outDistanceBufs[2] = {
    &outDistanceBuf1, &outDistanceBuf2};

  faiss::gpu::DeviceTensor<value_idx, 2, true> outIndexBuf1(
    gpu_res, faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream),
    {tileRows, numColTiles * k});
  faiss::gpu::DeviceTensor<value_idx, 2, true> outIndexBuf2(
    gpu_res, faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream),
    {tileRows, numColTiles * k});
  faiss::gpu::DeviceTensor<value_idx, 2, true> *outIndexBufs[2] = {
    &outIndexBuf1, &outIndexBuf2};

  auto streams = gpu_res->getAlternateStreamsCurrentDevice();
  faiss::gpu::streamWait(streams, {stream});

  int curStream = 0;
  bool interrupt = false;

  // Tile over the input queries
  for (int i = 0; i < m; i += tileRows) {
    if (interrupt || faiss::InterruptCallback::is_interrupted()) {
      interrupt = true;
      break;
    }

    int curQuerySize = std::min((size_t)tileRows, m - i);

    auto outDistanceView = out_dists_tensor.narrow(0, i, curQuerySize);
    auto outIndexView = out_inds_tensor.narrow(0, i, curQuerySize);

    auto queryView = x_tensor.narrow(0, i, curQuerySize);
    auto queryNormNiew = norms_tensor.narrow(0, i, curQuerySize);

    auto outDistanceBufRowView =
      outDistanceBufs[curStream]->narrow(0, 0, curQuerySize);
    auto outIndexBufRowView =
      outIndexBufs[curStream]->narrow(0, 0, curQuerySize);

    // Tile over the centroids
    for (int j = 0; j < m; j += tileCols) {
      if (faiss::InterruptCallback::is_interrupted()) {
        interrupt = true;
        break;
      }

      int curCentroidSize = std::min((size_t)tileCols, m - j);
      int curColTile = j / tileCols;

      auto centroidsView = sliceCentroids(x_tensor, true, j, curCentroidSize);

      auto distanceBufView = distanceBufs[curStream]
                               ->narrow(0, 0, curQuerySize)
                               .narrow(1, 0, curCentroidSize);

      auto outDistanceBufColView =
        outDistanceBufRowView.narrow(1, k * curColTile, k);
      auto outIndexBufColView = outIndexBufRowView.narrow(1, k * curColTile, k);

      runMatrixMult(distanceBufView,
                    false,  // not transposed
                    queryView,
                    false,  // transposed MM if col major
                    centroidsView,
                    true,  // transposed MM if row major
                    -2.0f, 0.0f, gpu_res->getBlasHandleCurrentDevice(),
                    streams[curStream]);

      if (tileCols == m) {
        // Write into the final output
        runL2SelectMin<value_t>(distanceBufView, norms_tensor,
                                core_dists_tensor, outDistanceView,
                                outIndexView, i, k, alpha, streams[curStream]);
      } else {
        auto centroidNormsView = norms_tensor.narrow(0, j, curCentroidSize);

        // Write into our intermediate output
        runL2SelectMin<value_t>(distanceBufView, norms_tensor,
                                core_dists_tensor, outDistanceBufColView,
                                outIndexBufColView, i, k, alpha,
                                streams[curStream]);
      }
    }

    // As we're finished with processing a full set of centroids, perform
    // the final k-selection
    if (tileCols != m) {
      // The indices are tile-relative; for each tile of k, we need to add
      // tileCols to the index
      faiss::gpu::runIncrementIndex(outIndexBufRowView, k, tileCols,
                                    streams[curStream]);

      faiss::gpu::runBlockSelectPair(outDistanceBufRowView, outIndexBufRowView,
                                     outDistanceView, outIndexView, false, k,
                                     streams[curStream]);
    }

    curStream = (curStream + 1) % 2;
  }

  // Have the desired ordering stream wait on the multi-stream
  faiss::gpu::streamWait({stream}, streams);

  if (interrupt) {
    FAISS_THROW_MSG("interrupted");
  }
}

/**
 * Extract core distances from KNN graph. This is essentially
 * performing a knn_dists[:,min_pts]
 * @tparam value_idx data type for integrals
 * @tparam value_t data type for distance
 * @tparam tpb block size for kernel
 * @param[in] knn_dists knn distance array (size n * k)
 * @param[in] k neighborhood size
 * @param[in] min_samples this neighbor will be selected for core distances
 * @param[in] n number of samples
 * @param[out] out output array (size n)
 * @param[in] stream stream for which to order cuda operations
 */
template <typename value_idx, typename value_t, int tpb = 256>
void core_distances(value_t *knn_dists, int k, int min_samples, size_t n,
                    value_t *out, cudaStream_t stream) {
  int blocks = raft::ceildiv(n, (size_t)tpb);

  auto exec_policy = rmm::exec_policy(stream);

  auto indices = thrust::make_counting_iterator<value_idx>(0);

  thrust::transform(exec_policy, indices, indices + n, out,
                    [=] __device__(value_idx row) {
                      return knn_dists[row * k + (min_samples - 1)];
                    });
}

/**
 * Constructs a mutual reachability graph, which is a k-nearest neighbors
 * graph projected into mutual reachability space using the following
 * function for each data point, where core_distance is the distance
 * to the kth neighbor: max(core_distance(a), core_distance(b), d(a, b))
 *
 * Unfortunately, points in the tails of the pdf (e.g. in sparse regions
 * of the space) can have very large neighborhoods, which will impact
 * nearby neighborhoods. Because of this, it's possible that the
 * radius for points in the main mass, which might have a very small
 * radius initially, to expand very large. As a result, the initial
 * knn which was used to compute the core distances may no longer
 * capture the actual neighborhoods after projection into mutual
 * reachability space.
 *
 * For the experimental version, we execute the knn twice- once
 * to compute the radii (core distances) and again to capture
 * the final neighborhoods. Future iterations of this algorithm
 * will work improve upon this "exact" version, by using
 * more specialized data structures, such as space-partitioning
 * structures. It has also been shown that approximate nearest
 * neighbors can yield reasonable neighborhoods as the
 * data sizes increase.
 *
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource reuse
 * @param[in] X input data points (size m * n)
 * @param[in] m number of rows in X
 * @param[in] n number of columns in X
 * @param[in] metric distance metric to use
 * @param[in] k neighborhood size
 * @param[in] min_samples this neighborhood will be selected for core distances
 * @param[in] alpha weight applied when internal distance is chosen for
 *            mutual reachability (value of 1.0 disables the weighting)
 * @param[out] indptr CSR indptr of output knn graph (size m + 1)
 * @param[out] core_dists output core distances array (size m)
 * @param[out] out COO object, uninitialized on entry, on exit it stores the
 *             (symmetrized) maximum reachability distance for the k nearest
 *             neighbors.
 */
template <typename value_idx, typename value_t>
void mutual_reachability_graph(const raft::handle_t &handle, const value_t *X,
                               size_t m, size_t n,
                               raft::distance::DistanceType metric, int k,
                               int min_samples, value_t alpha,
                               value_idx *indptr, value_t *core_dists,
                               raft::sparse::COO<value_t, value_idx> &out) {
  RAFT_EXPECTS(metric == raft::distance::DistanceType::L2SqrtExpanded,
               "Currently only L2 expanded distance is supported");

  auto stream = handle.get_stream();

  auto exec_policy = rmm::exec_policy(stream);

  std::vector<value_t *> inputs;
  inputs.push_back(const_cast<value_t *>(X));

  std::vector<int> sizes;
  sizes.push_back(m);

  // This is temporary. Once faiss is updated, we should be able to
  // pass value_idx through to knn.
  rmm::device_uvector<value_idx> coo_rows(k * m, stream);
  rmm::device_uvector<int64_t> int64_indices(k * m, stream);
  rmm::device_uvector<value_idx> inds(k * m, stream);
  rmm::device_uvector<value_t> dists(k * m, stream);

  // perform knn
  brute_force_knn(handle, inputs, sizes, n, const_cast<value_t *>(X), m,
                  int64_indices.data(), dists.data(), k, true, true, metric);

  // convert from current knn's 64-bit to 32-bit.
  thrust::transform(exec_policy, int64_indices.data(),
                    int64_indices.data() + int64_indices.size(), inds.data(),
                    [] __device__(int64_t in) -> value_idx { return in; });

  // Slice core distances (distances to kth nearest neighbor)
  core_distances<value_idx>(dists.data(), k, min_samples, m, core_dists,
                            stream);

  /**
   * Compute L2 norm
   */
  mutual_reachability_knn_l2(handle, inds.data(), dists.data(), X, m, n, k,
                             core_dists, (value_t)1.0 / alpha);

  printf("Creating coo rows\n");

  // self-loops get max distance
  auto coo_rows_counting_itr = thrust::make_counting_iterator<value_idx>(0);
  thrust::transform(exec_policy, coo_rows_counting_itr,
                    coo_rows_counting_itr + (m * k), coo_rows.data(),
                    [k] __device__(value_idx c) -> value_idx { return c / k; });

  CUDA_CHECK(cudaPeekAtLastError());

  printf("Done.\n");

  raft::sparse::linalg::symmetrize(handle, coo_rows.data(), inds.data(),
                                   dists.data(), m, m, k * m, out);

  raft::sparse::convert::sorted_coo_to_csr(
    out.rows(), out.nnz, indptr, m + 1, handle.get_device_allocator(), stream);

  // self-loops get max distance
  auto transform_in = thrust::make_zip_iterator(
    thrust::make_tuple(out.rows(), out.cols(), out.vals()));

  thrust::transform(
    exec_policy, transform_in, transform_in + out.nnz, out.vals(),
    [=] __device__(const thrust::tuple<value_idx, value_idx, value_t> &tup) {
      return thrust::get<0>(tup) == thrust::get<1>(tup)
               ? std::numeric_limits<value_t>::max()
               : thrust::get<2>(tup);
    });
}

};  // end namespace Reachability
};  // end namespace detail
};  // end namespace HDBSCAN
};  // end namespace ML