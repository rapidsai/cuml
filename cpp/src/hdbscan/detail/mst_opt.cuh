/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <cuml/cluster/hdbscan.hpp>
#include <cuml/common/utils.hpp>
#include <cuml/neighbors/knn.hpp>

#include <raft/core/host_device_accessor.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/distance/distance.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/neighbors/brute_force.cuh>
#include <raft/neighbors/detail/nn_descent.cuh>
#include <raft/neighbors/nn_descent_types.hpp>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/util/bitonic_sort.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <sys/time.h>

namespace NNDescent = raft::neighbors::experimental::nn_descent;

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Reachability {
// unnamed namespace to avoid multiple definition error
namespace {
inline double cur_time(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return ((double)tv.tv_sec + (double)tv.tv_usec * 1e-6);
}

template <typename T>
__device__ inline void swap(T& val1, T& val2)
{
  T val0 = val1;
  val1   = val2;
  val2   = val0;
}

template <typename K, typename V>
__device__ inline bool swap_if_needed(K& key1, K& key2, V& val1, V& val2, bool ascending)
{
  if (key1 == key2) { return false; }
  if ((key1 > key2) == ascending) {
    swap<K>(key1, key2);
    swap<V>(val1, val2);
    return true;
  }
  return false;
}

template <int MAX_DEGREE, class IdxT>
RAFT_KERNEL kern_prune(const IdxT* const knn_graph,  // [graph_chunk_size, graph_degree]
                       const uint32_t graph_size,
                       const uint32_t graph_degree,
                       const uint32_t degree,
                       const uint32_t batch_size,
                       const uint32_t batch_id,
                       uint8_t* const detour_count,          // [graph_chunk_size, graph_degree]
                       uint32_t* const num_no_detour_edges,  // [graph_size]
                       uint64_t* const stats)
{
  __shared__ uint32_t smem_num_detour[MAX_DEGREE];
  uint64_t* const num_retain = stats;
  uint64_t* const num_full   = stats + 1;

  const uint64_t nid = blockIdx.x + (batch_size * batch_id);
  if (nid >= graph_size) { return; }
  for (uint32_t k = threadIdx.x; k < graph_degree; k += blockDim.x) {
    smem_num_detour[k] = 0;
  }
  __syncthreads();

  const uint64_t iA = nid;
  if (iA >= graph_size) { return; }

  // count number of detours (A->D->B)
  for (uint32_t kAD = 0; kAD < graph_degree - 1; kAD++) {
    const uint64_t iD = knn_graph[kAD + (graph_degree * iA)];
    for (uint32_t kDB = threadIdx.x; kDB < graph_degree; kDB += blockDim.x) {
      const uint64_t iB_candidate = knn_graph[kDB + ((uint64_t)graph_degree * iD)];
      for (uint32_t kAB = kAD + 1; kAB < graph_degree; kAB++) {
        // if ( kDB < kAB )
        {
          const uint64_t iB = knn_graph[kAB + (graph_degree * iA)];
          if (iB == iB_candidate) {
            atomicAdd(smem_num_detour + kAB, 1);
            break;
          }
        }
      }
    }
    __syncthreads();
  }

  uint32_t num_edges_no_detour = 0;
  for (uint32_t k = threadIdx.x; k < graph_degree; k += blockDim.x) {
    detour_count[k + (graph_degree * iA)] = min(smem_num_detour[k], (uint32_t)255);
    if (smem_num_detour[k] == 0) { num_edges_no_detour++; }
  }
  num_edges_no_detour += __shfl_xor_sync(0xffffffff, num_edges_no_detour, 1);
  num_edges_no_detour += __shfl_xor_sync(0xffffffff, num_edges_no_detour, 2);
  num_edges_no_detour += __shfl_xor_sync(0xffffffff, num_edges_no_detour, 4);
  num_edges_no_detour += __shfl_xor_sync(0xffffffff, num_edges_no_detour, 8);
  num_edges_no_detour += __shfl_xor_sync(0xffffffff, num_edges_no_detour, 16);
  num_edges_no_detour = min(num_edges_no_detour, degree);

  if (threadIdx.x == 0) {
    num_no_detour_edges[iA] = num_edges_no_detour;
    atomicAdd((unsigned long long int*)num_retain, (unsigned long long int)num_edges_no_detour);
    if (num_edges_no_detour >= degree) { atomicAdd((unsigned long long int*)num_full, 1); }
  }
}

template <class IdxT>
RAFT_KERNEL kern_make_rev_graph(const IdxT* const dest_nodes,     // [graph_size]
                                IdxT* const rev_graph,            // [size, degree]
                                uint32_t* const rev_graph_count,  // [graph_size]
                                const uint32_t graph_size,
                                const uint32_t degree)
{
  const uint32_t tid  = threadIdx.x + (blockDim.x * blockIdx.x);
  const uint32_t tnum = blockDim.x * gridDim.x;

  for (uint32_t src_id = tid; src_id < graph_size; src_id += tnum) {
    const IdxT dest_id = dest_nodes[src_id];
    if (dest_id >= graph_size) continue;

    const uint32_t pos = atomicAdd(rev_graph_count + dest_id, 1);
    if (pos < degree) { rev_graph[pos + ((uint64_t)degree * dest_id)] = src_id; }
  }
}

template <class IdxT, class LabelT>
__device__ __host__ LabelT get_root_label(IdxT i, const LabelT* label)
{
  LabelT l = label[i];
  while (l != label[l]) {
    l = label[l];
  }
  return l;
}

template <class IdxT>
RAFT_KERNEL kern_mst_opt_update_graph(IdxT* mst_graph,                 // [graph_size, graph_degree]
                                      const IdxT* candidate_edges,     // [graph_size]
                                      IdxT* outgoing_num_edges,        // [graph_size]
                                      IdxT* incoming_num_edges,        // [graph_size]
                                      const IdxT* outgoing_max_edges,  // [graph_size]
                                      const IdxT* incoming_max_edges,  // [graph_size]
                                      const IdxT* label,               // [graph_size]
                                      const uint32_t graph_size,
                                      const uint32_t graph_degree,
                                      uint64_t* stats)
{
  const uint64_t i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i >= graph_size) return;

  int ret = 0;  // 0: No edge, 1: Direct edge, 2: Alternate edge, 3: Failure

  if (outgoing_num_edges[i] >= outgoing_max_edges[i]) return;
  uint64_t j = candidate_edges[i];
  if (j >= graph_size) return;
  const uint32_t ri = get_root_label(i, label);
  const uint32_t rj = get_root_label(j, label);
  if (ri == rj) return;

  // Try to add a direct edge to destination node with different label.
  if (incoming_num_edges[j] < incoming_max_edges[j]) {
    ret = 1;
    // Check to avoid duplication
    for (uint64_t kj = 0; kj < graph_degree; kj++) {
      uint64_t l = mst_graph[(graph_degree * j) + kj];
      if (l >= graph_size) continue;
      const uint32_t rl = get_root_label(l, label);
      if (ri == rl) {
        ret = 0;
        break;
      }
    }
    if (ret == 0) return;

    ret     = 0;
    auto kj = atomicAdd(incoming_num_edges + j, (IdxT)1);
    if (kj < incoming_max_edges[j]) {
      auto ki                                      = outgoing_num_edges[i]++;
      mst_graph[(graph_degree * (i)) + ki]         = j;  // outgoing
      mst_graph[(graph_degree * (j + 1)) - 1 - kj] = i;  // incoming
      ret                                          = 1;
    }
  }
  if (ret > 0) {
    atomicAdd((unsigned long long int*)stats + ret, 1);
    return;
  }

  // Try to add an edge to an alternate node instead
  ret = 3;
  for (uint64_t kj = 0; kj < graph_degree; kj++) {
    uint64_t l = mst_graph[(graph_degree * (j + 1)) - 1 - kj];
    if (l >= graph_size) continue;
    uint32_t rl = get_root_label(l, label);
    if (ri == rl) {
      ret = 0;
      break;
    }
    if (incoming_num_edges[l] >= incoming_max_edges[l]) continue;

    // Check to avoid duplication
    for (uint64_t kl = 0; kl < graph_degree; kl++) {
      uint64_t m = mst_graph[(graph_degree * l) + kl];
      if (m > graph_size) continue;
      uint32_t rm = get_root_label(m, label);
      if (ri == rm) {
        ret = 0;
        break;
      }
    }
    if (ret == 0) { break; }

    auto kl = atomicAdd(incoming_num_edges + l, (IdxT)1);
    if (kl < incoming_max_edges[l]) {
      auto ki                                      = outgoing_num_edges[i]++;
      mst_graph[(graph_degree * (i)) + ki]         = l;  // outgoing
      mst_graph[(graph_degree * (l + 1)) - 1 - kl] = i;  // incoming
      ret                                          = 2;
      break;
    }
  }
  if (ret > 0) { atomicAdd((unsigned long long int*)stats + ret, 1); }
}

template <class IdxT>
RAFT_KERNEL kern_mst_opt_labeling(IdxT* label,            // [graph_size]
                                  const IdxT* mst_graph,  // [graph_size, graph_degree]
                                  const uint32_t graph_size,
                                  const uint32_t graph_degree,
                                  uint64_t* stats)
{
  const uint64_t i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i >= graph_size) return;

  __shared__ uint32_t smem_updated[1];
  if (threadIdx.x == 0) { smem_updated[0] = 0; }
  __syncthreads();

  for (uint64_t ki = 0; ki < graph_degree; ki++) {
    uint64_t j = mst_graph[(graph_degree * i) + ki];
    if (j >= graph_size) continue;

    IdxT li = label[i];
    IdxT ri = get_root_label(i, label);
    if (ri < li) { atomicMin(label + i, ri); }
    IdxT lj = label[j];
    IdxT rj = get_root_label(j, label);
    if (rj < lj) { atomicMin(label + j, rj); }
    if (ri == rj) continue;

    if (ri > rj) {
      atomicCAS(label + i, ri, rj);
    } else if (rj > ri) {
      atomicCAS(label + j, rj, ri);
    }
    smem_updated[0] = 1;
  }

  __syncthreads();
  if ((threadIdx.x == 0) && (smem_updated[0] > 0)) { stats[0] = 1; }
}

template <class IdxT>
RAFT_KERNEL kern_mst_opt_cluster_size(IdxT* cluster_size,  // [graph_size]
                                      const IdxT* label,   // [graph_size]
                                      const uint32_t graph_size,
                                      uint64_t* stats)
{
  const uint64_t i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i >= graph_size) return;

  __shared__ uint64_t smem_num_clusters[1];
  if (threadIdx.x == 0) { smem_num_clusters[0] = 0; }
  __syncthreads();

  IdxT ri = get_root_label(i, label);
  if (ri == i) {
    atomicAdd((unsigned long long int*)smem_num_clusters, 1);
  } else {
    atomicAdd(cluster_size + ri, cluster_size[i]);
    cluster_size[i] = 0;
  }

  __syncthreads();
  if ((threadIdx.x == 0) && (smem_num_clusters[0] > 0)) {
    atomicAdd((unsigned long long int*)stats, (unsigned long long int)(smem_num_clusters[0]));
  }
}

template <class IdxT>
RAFT_KERNEL kern_mst_opt_postprocessing(IdxT* outgoing_num_edges,  // [graph_size]
                                        IdxT* incoming_num_edges,  // [graph_size]
                                        IdxT* outgoing_max_edges,  // [graph_size]
                                        IdxT* incoming_max_edges,  // [graph_size]
                                        const IdxT* cluster_size,  // [graph_size]
                                        const uint32_t graph_size,
                                        const uint32_t graph_degree,
                                        uint64_t* stats)
{
  const uint64_t i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i >= graph_size) return;

  __shared__ uint64_t smem_cluster_size_min[1];
  __shared__ uint64_t smem_cluster_size_max[1];
  __shared__ uint64_t smem_total_outgoing_edges[1];
  __shared__ uint64_t smem_total_incoming_edges[1];
  if (threadIdx.x == 0) {
    smem_cluster_size_min[0]     = stats[0];
    smem_cluster_size_max[0]     = stats[1];
    smem_total_outgoing_edges[0] = 0;
    smem_total_incoming_edges[0] = 0;
  }
  __syncthreads();

  // Adjust incoming_num_edges
  if (incoming_num_edges[i] > incoming_max_edges[i]) {
    incoming_num_edges[i] = incoming_max_edges[i];
  }

  // Calculate min/max of cluster_size
  if (cluster_size[i] > 0) {
    if (smem_cluster_size_min[0] > cluster_size[i]) {
      atomicMin((unsigned long long int*)smem_cluster_size_min,
                (unsigned long long int)(cluster_size[i]));
    }
    if (smem_cluster_size_max[0] < cluster_size[i]) {
      atomicMax((unsigned long long int*)smem_cluster_size_max,
                (unsigned long long int)(cluster_size[i]));
    }
  }

  // Calculate total number of outgoing/incoming edges
  atomicAdd((unsigned long long int*)smem_total_outgoing_edges,
            (unsigned long long int)(outgoing_num_edges[i]));
  atomicAdd((unsigned long long int*)smem_total_incoming_edges,
            (unsigned long long int)(incoming_num_edges[i]));

  // Adjust incoming/outgoing_max_edges
  if (outgoing_num_edges[i] == outgoing_max_edges[i]) {
    if (outgoing_num_edges[i] + incoming_num_edges[i] < graph_degree) {
      outgoing_max_edges[i] += 1;
      incoming_max_edges[i] -= 1;
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    atomicMin((unsigned long long int*)stats + 0,
              (unsigned long long int)(smem_cluster_size_min[0]));
    atomicMax((unsigned long long int*)stats + 1,
              (unsigned long long int)(smem_cluster_size_max[0]));
    atomicAdd((unsigned long long int*)stats + 2,
              (unsigned long long int)(smem_total_outgoing_edges[0]));
    atomicAdd((unsigned long long int*)stats + 3,
              (unsigned long long int)(smem_total_incoming_edges[0]));
  }
}

template <class T>
uint64_t pos_in_array(T val, const T* array, uint64_t num)
{
  for (uint64_t i = 0; i < num; i++) {
    if (val == array[i]) { return i; }
  }
  return num;
}

template <class T>
void shift_array(T* array, uint64_t num)
{
  for (uint64_t i = num; i > 0; i--) {
    array[i] = array[i - 1];
  }
}
}  // namespace

template <typename IdxT = uint32_t>
void mst_opt_update_graph(IdxT* mst_graph_ptr,
                          IdxT* candidate_edges_ptr,
                          IdxT* outgoing_num_edges_ptr,
                          IdxT* incoming_num_edges_ptr,
                          IdxT* outgoing_max_edges_ptr,
                          IdxT* incoming_max_edges_ptr,
                          IdxT* label_ptr,
                          IdxT graph_size,
                          uint32_t mst_graph_degree,
                          uint64_t k,
                          int& num_direct,
                          int& num_alternate,
                          int& num_failure)
{
#pragma omp parallel for reduction(+ : num_direct, num_alternate, num_failure)
  for (uint64_t ii = 0; ii < graph_size; ii++) {
    uint64_t i = ii;
    if (k % 2 == 0) { i = graph_size - (ii + 1); }
    int ret = 0;  // 0: No edge, 1: Direct edge, 2: Alternate edge, 3: Failure

    if (outgoing_num_edges_ptr[i] >= outgoing_max_edges_ptr[i]) continue;
    uint64_t j = candidate_edges_ptr[i];
    if (j >= graph_size) continue;
    if (label_ptr[i] == label_ptr[j]) continue;

    // Try to add a direct edge to destination node with different label.
    if (incoming_num_edges_ptr[j] < incoming_max_edges_ptr[j]) {
      ret = 1;
      // Check to avoid duplication
      for (uint64_t kj = 0; kj < mst_graph_degree; kj++) {
        uint64_t l = mst_graph_ptr[(mst_graph_degree * j) + kj];
        if (l >= graph_size) continue;
        if (label_ptr[i] == label_ptr[l]) {
          ret = 0;
          break;
        }
      }
      if (ret == 0) continue;

      // Use atomic to avoid conflicts, since 'incoming_num_edges_ptr[j]'
      // can be updated by other threads.
      ret = 0;
      uint32_t kj;
#pragma omp atomic capture
      kj = incoming_num_edges_ptr[j]++;
      if (kj < incoming_max_edges_ptr[j]) {
        auto ki                                              = outgoing_num_edges_ptr[i]++;
        mst_graph_ptr[(mst_graph_degree * (i)) + ki]         = j;  // OUT
        mst_graph_ptr[(mst_graph_degree * (j + 1)) - 1 - kj] = i;  // IN
        ret                                                  = 1;
      }
    }
    if (ret == 1) {
      num_direct += 1;
      continue;
    }

    // Try to add an edge to an alternate node instead
    ret = 3;
    for (uint64_t kj = 0; kj < mst_graph_degree; kj++) {
      uint64_t l = mst_graph_ptr[(mst_graph_degree * (j + 1)) - 1 - kj];
      if (l >= graph_size) continue;
      if (label_ptr[i] == label_ptr[l]) {
        ret = 0;
        break;
      }
      if (incoming_num_edges_ptr[l] >= incoming_max_edges_ptr[l]) continue;

      // Check to avoid duplication
      for (uint64_t kl = 0; kl < mst_graph_degree; kl++) {
        uint64_t m = mst_graph_ptr[(mst_graph_degree * l) + kl];
        if (m > graph_size) continue;
        if (label_ptr[i] == label_ptr[m]) {
          ret = 0;
          break;
        }
      }
      if (ret == 0) { break; }

      // Use atomic to avoid conflicts, since 'incoming_num_edges_ptr[l]'
      // can be updated by other threads.
      uint32_t kl;
#pragma omp atomic capture
      kl = incoming_num_edges_ptr[l]++;
      if (kl < incoming_max_edges_ptr[l]) {
        auto ki                                              = outgoing_num_edges_ptr[i]++;
        mst_graph_ptr[(mst_graph_degree * (i)) + ki]         = l;  // OUT
        mst_graph_ptr[(mst_graph_degree * (l + 1)) - 1 - kl] = i;  // IN
        ret                                                  = 2;
        break;
      }
    }
    if (ret == 2) {
      num_alternate += 1;
    } else if (ret == 3) {
      num_failure += 1;
    }
  }
}

//
// Create approximate MSTs with kNN graphs as input to guarantee connectivity of search graphs
//
// * Since there is an upper limit to the degree of a graph for search, what is created is a
//   degree-constraied MST.
// * The number of edges is not a minimum because strict MST is not required. Therefore, it is
//   an approximate MST.
// * If the input kNN graph is disconnected, random connection is added to the largest cluster.
//
template <typename IdxT = uint32_t>
void mst_optimization(raft::resources const& res,
                      raft::host_matrix_view<IdxT, int64_t, raft::row_major> input_graph,
                      raft::host_matrix_view<IdxT, int64_t, raft::row_major> output_graph,
                      raft::host_vector_view<uint32_t, int64_t> mst_graph_num_edges,
                      bool use_gpu = true)
{
  const double time_mst_opt_start = cur_time();

  const IdxT graph_size              = input_graph.extent(0);
  const uint32_t input_graph_degree  = input_graph.extent(1);
  const uint32_t output_graph_degree = output_graph.extent(1);
  auto input_graph_ptr               = input_graph.data_handle();
  auto output_graph_ptr              = output_graph.data_handle();
  auto mst_graph_num_edges_ptr       = mst_graph_num_edges.data_handle();

  // Allocate temporal arrays
  const uint32_t mst_graph_degree = output_graph_degree;
  auto mst_graph              = raft::make_host_matrix<IdxT, int64_t>(graph_size, mst_graph_degree);
  auto outgoing_max_edges     = raft::make_host_vector<IdxT, int64_t>(graph_size);
  auto incoming_max_edges     = raft::make_host_vector<IdxT, int64_t>(graph_size);
  auto outgoing_num_edges     = raft::make_host_vector<IdxT, int64_t>(graph_size);
  auto incoming_num_edges     = raft::make_host_vector<IdxT, int64_t>(graph_size);
  auto label                  = raft::make_host_vector<IdxT, int64_t>(graph_size);
  auto cluster_size           = raft::make_host_vector<IdxT, int64_t>(graph_size);
  auto candidate_edges        = raft::make_host_vector<IdxT, int64_t>(graph_size);
  auto mst_graph_ptr          = mst_graph.data_handle();
  auto outgoing_max_edges_ptr = outgoing_max_edges.data_handle();
  auto incoming_max_edges_ptr = incoming_max_edges.data_handle();
  auto outgoing_num_edges_ptr = outgoing_num_edges.data_handle();
  auto incoming_num_edges_ptr = incoming_num_edges.data_handle();
  auto label_ptr              = label.data_handle();
  auto cluster_size_ptr       = cluster_size.data_handle();
  auto candidate_edges_ptr    = candidate_edges.data_handle();

  // Initialize arrays
#pragma omp parallel for
  for (uint64_t i = 0; i < graph_size; i++) {
    for (uint64_t k = 0; k < mst_graph_degree; k++) {
      // mst_graph_ptr[(mst_graph_degree * i) + k] = graph_size;
      mst_graph(i, k) = graph_size;
    }
    outgoing_max_edges_ptr[i] = 2;
    incoming_max_edges_ptr[i] = mst_graph_degree - outgoing_max_edges_ptr[i];
    outgoing_num_edges_ptr[i] = 0;
    incoming_num_edges_ptr[i] = 0;
    label_ptr[i]              = i;
    cluster_size_ptr[i]       = 1;
  }

  // Allocate arrays on GPU
  uint32_t d_graph_size = graph_size;
  if (!use_gpu) {
    // (*) If GPU is not used, arrays of size 0 are created.
    d_graph_size = 0;
  }
  auto d_mst_graph_num_edges = raft::make_device_vector<IdxT, int64_t>(res, d_graph_size);
  auto d_mst_graph = raft::make_device_matrix<IdxT, int64_t>(res, d_graph_size, mst_graph_degree);
  auto d_outgoing_max_edges      = raft::make_device_vector<IdxT, int64_t>(res, d_graph_size);
  auto d_incoming_max_edges      = raft::make_device_vector<IdxT, int64_t>(res, d_graph_size);
  auto d_outgoing_num_edges      = raft::make_device_vector<IdxT, int64_t>(res, d_graph_size);
  auto d_incoming_num_edges      = raft::make_device_vector<IdxT, int64_t>(res, d_graph_size);
  auto d_label                   = raft::make_device_vector<IdxT, int64_t>(res, d_graph_size);
  auto d_cluster_size            = raft::make_device_vector<IdxT, int64_t>(res, d_graph_size);
  auto d_candidate_edges         = raft::make_device_vector<IdxT, int64_t>(res, d_graph_size);
  auto d_mst_graph_num_edges_ptr = d_mst_graph_num_edges.data_handle();
  auto d_mst_graph_ptr           = d_mst_graph.data_handle();
  auto d_outgoing_max_edges_ptr  = d_outgoing_max_edges.data_handle();
  auto d_incoming_max_edges_ptr  = d_incoming_max_edges.data_handle();
  auto d_outgoing_num_edges_ptr  = d_outgoing_num_edges.data_handle();
  auto d_incoming_num_edges_ptr  = d_incoming_num_edges.data_handle();
  auto d_label_ptr               = d_label.data_handle();
  auto d_cluster_size_ptr        = d_cluster_size.data_handle();
  auto d_candidate_edges_ptr     = d_candidate_edges.data_handle();

  constexpr int stats_size = 4;
  auto stats               = raft::make_host_vector<uint64_t, int64_t>(stats_size);
  auto d_stats             = raft::make_device_vector<uint64_t, int64_t>(res, stats_size);
  auto stats_ptr           = stats.data_handle();
  auto d_stats_ptr         = d_stats.data_handle();

  if (use_gpu) {
    raft::copy(d_mst_graph_ptr,
               mst_graph_ptr,
               (size_t)graph_size * mst_graph_degree,
               raft::resource::get_cuda_stream(res));
    raft::copy(d_outgoing_num_edges_ptr,
               outgoing_num_edges_ptr,
               (size_t)graph_size,
               raft::resource::get_cuda_stream(res));
    raft::copy(d_incoming_num_edges_ptr,
               incoming_num_edges_ptr,
               (size_t)graph_size,
               raft::resource::get_cuda_stream(res));
    raft::copy(d_outgoing_max_edges_ptr,
               outgoing_max_edges_ptr,
               (size_t)graph_size,
               raft::resource::get_cuda_stream(res));
    raft::copy(d_incoming_max_edges_ptr,
               incoming_max_edges_ptr,
               (size_t)graph_size,
               raft::resource::get_cuda_stream(res));
    raft::copy(d_label_ptr, label_ptr, (size_t)graph_size, raft::resource::get_cuda_stream(res));
    raft::copy(d_cluster_size_ptr,
               cluster_size_ptr,
               (size_t)graph_size,
               raft::resource::get_cuda_stream(res));
  }

  IdxT num_clusters     = 0;
  IdxT num_clusters_pre = graph_size;
  IdxT cluster_size_min = graph_size;
  IdxT cluster_size_max = 0;
  for (uint64_t k = 0; k <= input_graph_degree; k++) {
    int num_direct    = 0;
    int num_alternate = 0;
    int num_failure   = 0;

    // 1. Prepare candidate edges
    if (k == input_graph_degree) {
      // If the number of clusters does not converge to 1, then edges are
      // made from all nodes not belonging to the main cluster to any node
      // in the main cluster.
      raft::copy(cluster_size_ptr,
                 d_cluster_size_ptr,
                 (size_t)graph_size,
                 raft::resource::get_cuda_stream(res));
      raft::copy(label_ptr, d_label_ptr, (size_t)graph_size, raft::resource::get_cuda_stream(res));
      raft::resource::sync_stream(res);
      uint32_t main_cluster_label = graph_size;
#pragma omp parallel for reduction(min : main_cluster_label)
      for (uint64_t i = 0; i < graph_size; i++) {
        if ((cluster_size_ptr[i] == cluster_size_max) && (main_cluster_label > i)) {
          main_cluster_label = i;
        }
      }
#pragma omp parallel for
      for (uint64_t i = 0; i < graph_size; i++) {
        candidate_edges_ptr[i] = graph_size;
        if (label_ptr[i] == main_cluster_label) continue;
        uint64_t j = i;
        while (label_ptr[j] != main_cluster_label) {
          constexpr uint32_t ofst = 97;
          j                       = (j + ofst) % graph_size;
        }
        candidate_edges_ptr[i] = j;
      }
    } else {
      // Copy rank-k edges from the input knn graph to 'candidate_edges'
#pragma omp parallel for
      for (uint64_t i = 0; i < graph_size; i++) {
        candidate_edges_ptr[i] = input_graph_ptr[k + (input_graph_degree * i)];
      }
    }
    // 2. Update MST graph
    //  * Try to add candidate edges to MST graph
    if (use_gpu) {
      raft::copy(d_candidate_edges_ptr,
                 candidate_edges_ptr,
                 graph_size,
                 raft::resource::get_cuda_stream(res));
      stats_ptr[0] = 0;
      stats_ptr[1] = num_direct;
      stats_ptr[2] = num_alternate;
      stats_ptr[3] = num_failure;
      raft::copy(d_stats_ptr, stats_ptr, 4, raft::resource::get_cuda_stream(res));
      constexpr uint64_t n_threads = 256;
      const dim3 threads(n_threads, 1, 1);
      const dim3 blocks(raft::ceildiv<uint64_t>(graph_size, n_threads), 1, 1);
      kern_mst_opt_update_graph<<<blocks, threads, 0, raft::resource::get_cuda_stream(res)>>>(
        d_mst_graph_ptr,
        d_candidate_edges_ptr,
        d_outgoing_num_edges_ptr,
        d_incoming_num_edges_ptr,
        d_outgoing_max_edges_ptr,
        d_incoming_max_edges_ptr,
        d_label_ptr,
        graph_size,
        mst_graph_degree,
        d_stats_ptr);

      raft::copy(stats_ptr, d_stats_ptr, 4, raft::resource::get_cuda_stream(res));
      raft::resource::sync_stream(res);
      num_direct    = stats_ptr[1];
      num_alternate = stats_ptr[2];
      num_failure   = stats_ptr[3];
    } else {
      mst_opt_update_graph(mst_graph_ptr,
                           candidate_edges_ptr,
                           outgoing_num_edges_ptr,
                           incoming_num_edges_ptr,
                           outgoing_max_edges_ptr,
                           incoming_max_edges_ptr,
                           label_ptr,
                           graph_size,
                           mst_graph_degree,
                           k,
                           num_direct,
                           num_alternate,
                           num_failure);
    }
    // 3. Labeling
    uint32_t flag_update = 1;
    while (flag_update) {
      flag_update = 0;
      if (use_gpu) {
        stats_ptr[0] = flag_update;
        raft::copy(d_stats_ptr, stats_ptr, 1, raft::resource::get_cuda_stream(res));

        constexpr uint64_t n_threads = 256;
        const dim3 threads(n_threads, 1, 1);
        const dim3 blocks((graph_size + n_threads - 1) / n_threads, 1, 1);
        kern_mst_opt_labeling<<<blocks, threads, 0, raft::resource::get_cuda_stream(res)>>>(
          d_label_ptr, d_mst_graph_ptr, graph_size, mst_graph_degree, d_stats_ptr);

        raft::copy(stats_ptr, d_stats_ptr, 1, raft::resource::get_cuda_stream(res));
        raft::resource::sync_stream(res);
        flag_update = stats_ptr[0];
      } else {
#pragma omp parallel for reduction(+ : flag_update)
        for (uint64_t i = 0; i < graph_size; i++) {
          for (uint64_t ki = 0; ki < mst_graph_degree; ki++) {
            uint64_t j = mst_graph_ptr[(mst_graph_degree * i) + ki];
            if (j >= graph_size) continue;
            if (label_ptr[i] > label_ptr[j]) {
              flag_update += 1;
              label_ptr[i] = label_ptr[j];
            }
          }
        }
      }
    }
    // 4. Calculate the number of clusters and the size of each cluster
    num_clusters = 0;
    if (use_gpu) {
      stats_ptr[0] = num_clusters;
      raft::copy(d_stats_ptr, stats_ptr, 1, raft::resource::get_cuda_stream(res));

      constexpr uint64_t n_threads = 256;
      const dim3 threads(n_threads, 1, 1);
      const dim3 blocks(raft::ceildiv<uint64_t>(graph_size, n_threads), 1, 1);
      kern_mst_opt_cluster_size<<<blocks, threads, 0, raft::resource::get_cuda_stream(res)>>>(
        d_cluster_size_ptr, d_label_ptr, graph_size, d_stats_ptr);

      raft::copy(stats_ptr, d_stats_ptr, 1, raft::resource::get_cuda_stream(res));
      raft::resource::sync_stream(res);
      num_clusters = stats_ptr[0];
    } else {
#pragma omp parallel for reduction(+ : num_clusters)
      for (uint64_t i = 0; i < graph_size; i++) {
        uint64_t ri = get_root_label(i, label_ptr);
        if (ri == i) {
          num_clusters += 1;
        } else {
#pragma omp atomic update
          cluster_size_ptr[ri] += cluster_size_ptr[i];
          cluster_size_ptr[i] = 0;
        }
      }
    }
    // 5. Postprocessings
    //  * Adjust incoming_num_edges
    //  * Calculate the min/max size of clusters.
    //  * Calculate the total number of outgoing/incoming edges
    //  * Increase the limit of outgoing edges as needed
    cluster_size_min              = graph_size;
    cluster_size_max              = 0;
    uint64_t total_outgoing_edges = 0;
    uint64_t total_incoming_edges = 0;
    if (use_gpu) {
      stats_ptr[0] = cluster_size_min;
      stats_ptr[1] = cluster_size_max;
      stats_ptr[2] = total_outgoing_edges;
      stats_ptr[3] = total_incoming_edges;
      raft::copy(d_stats_ptr, stats_ptr, 4, raft::resource::get_cuda_stream(res));

      constexpr uint64_t n_threads = 256;
      const dim3 threads(n_threads, 1, 1);
      const dim3 blocks((graph_size + n_threads - 1) / n_threads, 1, 1);
      kern_mst_opt_postprocessing<<<blocks, threads, 0, raft::resource::get_cuda_stream(res)>>>(
        d_outgoing_num_edges_ptr,
        d_incoming_num_edges_ptr,
        d_outgoing_max_edges_ptr,
        d_incoming_max_edges_ptr,
        d_cluster_size_ptr,
        graph_size,
        mst_graph_degree,
        d_stats_ptr);

      raft::copy(stats_ptr, d_stats_ptr, 4, raft::resource::get_cuda_stream(res));
      raft::resource::sync_stream(res);
      cluster_size_min     = stats_ptr[0];
      cluster_size_max     = stats_ptr[1];
      total_outgoing_edges = stats_ptr[2];
      total_incoming_edges = stats_ptr[3];
    } else {
#pragma omp parallel for
      for (uint64_t i = 0; i < graph_size; i++) {
        if (incoming_num_edges_ptr[i] > incoming_max_edges_ptr[i]) {
          incoming_num_edges_ptr[i] = incoming_max_edges_ptr[i];
        }
      }

#pragma omp parallel for reduction(max : cluster_size_max) reduction(min : cluster_size_min)
      for (uint64_t i = 0; i < graph_size; i++) {
        if (cluster_size_ptr[i] == 0) continue;
        cluster_size_min = min(cluster_size_min, cluster_size_ptr[i]);
        cluster_size_max = max(cluster_size_max, cluster_size_ptr[i]);
      }

#pragma omp parallel for reduction(+ : total_outgoing_edges, total_incoming_edges)
      for (uint64_t i = 0; i < graph_size; i++) {
        total_outgoing_edges += outgoing_num_edges_ptr[i];
        total_incoming_edges += incoming_num_edges_ptr[i];
      }

#pragma omp parallel for
      for (uint64_t i = 0; i < graph_size; i++) {
        if (outgoing_num_edges_ptr[i] < outgoing_max_edges_ptr[i]) continue;
        if (outgoing_num_edges_ptr[i] + incoming_num_edges_ptr[i] == mst_graph_degree) continue;
        assert(outgoing_num_edges_ptr[i] + incoming_num_edges_ptr[i] < mst_graph_degree);
        outgoing_max_edges_ptr[i] += 1;
        incoming_max_edges_ptr[i] = mst_graph_degree - outgoing_max_edges_ptr[i];
      }
    }
    // 6. Show stats
    if (num_clusters != num_clusters_pre) {
      std::string msg = "# k: " + std::to_string(k);
      msg += ", num_clusters: " + std::to_string(num_clusters);
      msg += ", cluster_size: " + std::to_string(cluster_size_min) + " to " +
             std::to_string(cluster_size_max);
      msg += ", total_num_edges: " + std::to_string(total_outgoing_edges) + ", " +
             std::to_string(total_incoming_edges);
      if (num_alternate + num_failure > 0) {
        msg += ", altenate: " + std::to_string(num_alternate);
        if (num_failure > 0) { msg += ", failure: " + std::to_string(num_failure); }
      }
    }
    assert(num_clusters > 0);
    assert(total_outgoing_edges == total_incoming_edges);
    if (num_clusters == 1) { break; }
    num_clusters_pre = num_clusters;
  }
  // The edges that make up the MST are stored as edges in the output graph.
  if (use_gpu) {
    raft::copy(mst_graph_ptr,
               d_mst_graph_ptr,
               (size_t)graph_size * mst_graph_degree,
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
  }
#pragma omp parallel for
  for (uint64_t i = 0; i < graph_size; i++) {
    uint64_t k = 0;
    for (uint64_t kj = 0; kj < mst_graph_degree; kj++) {
      uint64_t j = mst_graph_ptr[(mst_graph_degree * i) + kj];
      if (j >= graph_size) continue;

      // Check to avoid duplication
      auto flag_match = false;
      for (uint64_t ki = 0; ki < k; ki++) {
        if (j == output_graph_ptr[(output_graph_degree * i) + ki]) {
          flag_match = true;
          break;
        }
      }
      if (flag_match) continue;

      output_graph_ptr[(output_graph_degree * i) + k] = j;
      k += 1;
    }
    mst_graph_num_edges_ptr[i] = k;
  }

  const double time_mst_opt_end = cur_time();
}

template <
  typename IdxT = uint32_t,
  typename g_accessor =
    raft::host_device_accessor<std::experimental::default_accessor<IdxT>, raft::memory_type::host>>
void optimize(raft::resources const& res,
              raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
              raft::host_matrix_view<IdxT, int64_t, raft::row_major> new_graph,
              const bool guarantee_connectivity = true)
{
  auto large_tmp_mr = raft::resource::get_large_workspace_resource(res);

  RAFT_EXPECTS(knn_graph.extent(0) == new_graph.extent(0),
               "Each input array is expected to have the same number of rows");
  RAFT_EXPECTS(new_graph.extent(1) <= knn_graph.extent(1),
               "output graph cannot have more columns than input graph");
  const uint32_t input_graph_degree  = knn_graph.extent(1);
  const uint32_t output_graph_degree = new_graph.extent(1);
  auto input_graph_ptr               = knn_graph.data_handle();
  auto output_graph_ptr              = new_graph.data_handle();
  const IdxT graph_size              = new_graph.extent(0);

  // MST optimization
  auto mst_graph_num_edges     = raft::make_host_vector<uint32_t, int64_t>(graph_size);
  auto mst_graph_num_edges_ptr = mst_graph_num_edges.data_handle();
#pragma omp parallel for
  for (uint64_t i = 0; i < graph_size; i++) {
    mst_graph_num_edges_ptr[i] = 0;
  }
  if (guarantee_connectivity) {
    constexpr bool use_gpu = true;
    mst_optimization(res, knn_graph, new_graph, mst_graph_num_edges.view(), use_gpu);
  }

  auto pruned_graph = raft::make_host_matrix<uint32_t, int64_t>(graph_size, output_graph_degree);
  {
    //
    // Prune kNN graph
    //
    auto d_detour_count = raft::make_device_mdarray<uint8_t>(
      res, large_tmp_mr, raft::make_extents<int64_t>(graph_size, input_graph_degree));

    RAFT_CUDA_TRY(cudaMemsetAsync(d_detour_count.data_handle(),
                                  0xff,
                                  graph_size * input_graph_degree * sizeof(uint8_t),
                                  raft::resource::get_cuda_stream(res)));

    auto d_num_no_detour_edges = raft::make_device_mdarray<uint32_t>(
      res, large_tmp_mr, raft::make_extents<int64_t>(graph_size));
    RAFT_CUDA_TRY(cudaMemsetAsync(d_num_no_detour_edges.data_handle(),
                                  0x00,
                                  graph_size * sizeof(uint32_t),
                                  raft::resource::get_cuda_stream(res)));

    auto dev_stats  = raft::make_device_vector<uint64_t>(res, 2);
    auto host_stats = raft::make_host_vector<uint64_t>(2);

    //
    // Prune unimportant edges.
    //
    // The edge to be retained is determined without explicitly considering
    // distance or angle. Suppose the edge is the k-th edge of some node-A to
    // node-B (A->B). Among the edges originating at node-A, there are k-1 edges
    // shorter than the edge A->B. Each of these k-1 edges are connected to a
    // different k-1 nodes. Among these k-1 nodes, count the number of nodes with
    // edges to node-B, which is the number of 2-hop detours for the edge A->B.
    // Once the number of 2-hop detours has been counted for all edges, the
    // specified number of edges are picked up for each node, starting with the
    // edge with the lowest number of 2-hop detours.
    //
    const double time_prune_start = cur_time();

    // Copy input_graph_ptr over to device if necessary
    auto d_input_graph =
      raft::make_device_matrix<IdxT, int64_t>(res, graph_size, input_graph_degree);
    raft::copy(d_input_graph.data_handle(),
               input_graph_ptr,
               graph_size * input_graph_degree,
               raft::resource::get_cuda_stream(res));
    // device_matrix_view_from_host d_input_graph(
    //   res,
    //   raft::make_host_matrix_view<IdxT, int64_t>(input_graph_ptr, graph_size,
    //   input_graph_degree));

    constexpr int MAX_DEGREE = 1024;
    if (input_graph_degree > MAX_DEGREE) {
      RAFT_FAIL(
        "The degree of input knn graph is too large (%u). "
        "It must be equal to or smaller than %d.",
        input_graph_degree,
        1024);
    }
    const uint32_t batch_size =
      std::min(static_cast<uint32_t>(graph_size), static_cast<uint32_t>(256 * 1024));
    const uint32_t num_batch = (graph_size + batch_size - 1) / batch_size;
    const dim3 threads_prune(32, 1, 1);
    const dim3 blocks_prune(batch_size, 1, 1);

    RAFT_CUDA_TRY(cudaMemsetAsync(
      dev_stats.data_handle(), 0, sizeof(uint64_t) * 2, raft::resource::get_cuda_stream(res)));

    for (uint32_t i_batch = 0; i_batch < num_batch; i_batch++) {
      kern_prune<MAX_DEGREE, IdxT>
        <<<blocks_prune, threads_prune, 0, raft::resource::get_cuda_stream(res)>>>(
          d_input_graph.data_handle(),
          graph_size,
          input_graph_degree,
          output_graph_degree,
          batch_size,
          i_batch,
          d_detour_count.data_handle(),
          d_num_no_detour_edges.data_handle(),
          dev_stats.data_handle());
      raft::resource::sync_stream(res);
    }
    raft::resource::sync_stream(res);

    // host_matrix_view_from_device<uint8_t, int64_t> detour_count(res, d_detour_count.view());
    auto detour_count = raft::make_host_matrix<uint8_t, int64_t>(graph_size, input_graph_degree);
    raft::copy(detour_count.data_handle(),
               d_detour_count.data_handle(),
               graph_size * input_graph_degree,
               raft::resource::get_cuda_stream(res));

    raft::copy(
      host_stats.data_handle(), dev_stats.data_handle(), 2, raft::resource::get_cuda_stream(res));
    const auto num_keep = host_stats.data_handle()[0];
    const auto num_full = host_stats.data_handle()[1];

    // Create pruned kNN graph
#pragma omp parallel for
    for (uint64_t i = 0; i < graph_size; i++) {
      // Find the `output_graph_degree` smallest detourable count nodes by checking the detourable
      // count of the neighbors while increasing the target detourable count from zero.
      uint64_t pk         = 0;
      uint32_t num_detour = 0;
      while (pk < output_graph_degree) {
        uint32_t next_num_detour = std::numeric_limits<uint32_t>::max();
        for (uint64_t k = 0; k < input_graph_degree; k++) {
          const auto num_detour_k = detour_count.data_handle()[k + (input_graph_degree * i)];
          // Find the detourable count to check in the next iteration
          if (num_detour_k > num_detour) {
            next_num_detour = std::min(static_cast<uint32_t>(num_detour_k), next_num_detour);
          }

          // Store the neighbor index if its detourable count is equal to `num_detour`.
          if (num_detour_k != num_detour) { continue; }
          output_graph_ptr[pk + (output_graph_degree * i)] =
            input_graph_ptr[k + (input_graph_degree * i)];
          pk += 1;
          if (pk >= output_graph_degree) break;
        }
        if (pk >= output_graph_degree) break;

        assert(next_num_detour != std::numeric_limits<uint32_t>::max());
        num_detour = next_num_detour;
      }
      RAFT_EXPECTS(pk == output_graph_degree,
                   "Couldn't find the output_graph_degree (%u) smallest detourable count nodes for "
                   "node %lu in the rank-based node reranking process",
                   output_graph_degree,
                   static_cast<uint64_t>(i));
    }

    const double time_prune_end = cur_time();
  }

  auto rev_graph       = raft::make_host_matrix<IdxT, int64_t>(graph_size, output_graph_degree);
  auto rev_graph_count = raft::make_host_vector<uint32_t, int64_t>(graph_size);

  {
    //
    // Make reverse graph
    //
    const double time_make_start = cur_time();
    auto d_rev_graph =
      raft::make_device_matrix<IdxT, int64_t>(res, graph_size, output_graph_degree);
    raft::copy(d_rev_graph.data_handle(),
               rev_graph.data_handle(),
               graph_size * output_graph_degree,
               raft::resource::get_cuda_stream(res));
    // device_matrix_view_from_host<IdxT, int64_t> d_rev_graph(res, rev_graph.view());

    RAFT_CUDA_TRY(cudaMemsetAsync(d_rev_graph.data_handle(),
                                  0xff,
                                  graph_size * output_graph_degree * sizeof(IdxT),
                                  raft::resource::get_cuda_stream(res)));

    auto d_rev_graph_count = raft::make_device_mdarray<uint32_t>(
      res, large_tmp_mr, raft::make_extents<int64_t>(graph_size));
    RAFT_CUDA_TRY(cudaMemsetAsync(d_rev_graph_count.data_handle(),
                                  0x00,
                                  graph_size * sizeof(uint32_t),
                                  raft::resource::get_cuda_stream(res)));

    auto dest_nodes = raft::make_host_vector<IdxT, int64_t>(graph_size);
    auto d_dest_nodes =
      raft::make_device_mdarray<IdxT>(res, large_tmp_mr, raft::make_extents<int64_t>(graph_size));

    for (uint64_t k = 0; k < output_graph_degree; k++) {
#pragma omp parallel for
      for (uint64_t i = 0; i < graph_size; i++) {
        // dest_nodes.data_handle()[i] = output_graph_ptr[k + (output_graph_degree * i)];
        dest_nodes(i) = output_graph_ptr[k + (output_graph_degree * i)];
      }
      raft::resource::sync_stream(res);

      raft::copy(d_dest_nodes.data_handle(),
                 dest_nodes.data_handle(),
                 graph_size,
                 raft::resource::get_cuda_stream(res));

      dim3 threads(256, 1, 1);
      dim3 blocks(1024, 1, 1);
      kern_make_rev_graph<<<blocks, threads, 0, raft::resource::get_cuda_stream(res)>>>(
        d_dest_nodes.data_handle(),
        d_rev_graph.data_handle(),
        d_rev_graph_count.data_handle(),
        graph_size,
        output_graph_degree);
    }

    raft::resource::sync_stream(res);

    raft::copy(rev_graph.data_handle(),
               d_rev_graph.data_handle(),
               graph_size * output_graph_degree,
               raft::resource::get_cuda_stream(res));
    raft::copy(rev_graph_count.data_handle(),
               d_rev_graph_count.data_handle(),
               graph_size,
               raft::resource::get_cuda_stream(res));

    const double time_make_end = cur_time();
  }

  {
    //
    // Create search graphs from MST and pruned and reverse graphs
    //
    const double time_replace_start = cur_time();

#pragma omp parallel for
    for (uint64_t i = 0; i < graph_size; i++) {
      auto my_fwd_graph = pruned_graph.data_handle() + (output_graph_degree * i);
      auto my_rev_graph = rev_graph.data_handle() + (output_graph_degree * i);
      auto my_out_graph = output_graph_ptr + (output_graph_degree * i);
      uint32_t kf       = 0;
      uint32_t k        = mst_graph_num_edges_ptr[i];

      const uint64_t num_protected_edges = max(k, output_graph_degree / 2);
      assert(num_protected_edges <= output_graph_degree);
      if (num_protected_edges == output_graph_degree) continue;

      // Append edges from the pruned graph to output graph
      while (k < output_graph_degree && kf < output_graph_degree) {
        if (my_fwd_graph[kf] < graph_size) {
          auto flag_match = false;
          for (uint32_t kk = 0; kk < k; kk++) {
            if (my_out_graph[kk] == my_fwd_graph[kf]) {
              flag_match = true;
              break;
            }
          }
          if (!flag_match) {
            my_out_graph[k] = my_fwd_graph[kf];
            k += 1;
          }
        }
        kf += 1;
      }
      assert(k == output_graph_degree);
      assert(kf <= output_graph_degree);

      // Replace some edges of the output graph with edges of the reverse graph.
      uint32_t kr = std::min(rev_graph_count.data_handle()[i], output_graph_degree);
      while (kr) {
        kr -= 1;
        if (my_rev_graph[kr] < graph_size) {
          uint64_t pos = pos_in_array<IdxT>(my_rev_graph[kr], my_out_graph, output_graph_degree);
          if (pos < num_protected_edges) { continue; }
          uint64_t num_shift = pos - num_protected_edges;
          if (pos >= output_graph_degree) {
            num_shift = output_graph_degree - num_protected_edges - 1;
          }
          shift_array<IdxT>(my_out_graph + num_protected_edges, num_shift);
          my_out_graph[num_protected_edges] = my_rev_graph[kr];
        }
      }
    }

    const double time_replace_end = cur_time();

    /* stats */
    uint64_t num_replaced_edges = 0;
#pragma omp parallel for reduction(+ : num_replaced_edges)
    for (uint64_t i = 0; i < graph_size; i++) {
      for (uint64_t k = 0; k < output_graph_degree; k++) {
        const uint64_t j = output_graph_ptr[k + (output_graph_degree * i)];
        const uint64_t pos =
          pos_in_array<IdxT>(j, output_graph_ptr + (output_graph_degree * i), output_graph_degree);
        if (pos == output_graph_degree) { num_replaced_edges += 1; }
      }
    }
  }

  // Check number of incoming edges
  {
    auto in_edge_count     = raft::make_host_vector<uint32_t, int64_t>(graph_size);
    auto in_edge_count_ptr = in_edge_count.data_handle();
#pragma omp parallel for
    for (uint64_t i = 0; i < graph_size; i++) {
      in_edge_count_ptr[i] = 0;
    }
#pragma omp parallel for
    for (uint64_t i = 0; i < graph_size; i++) {
      for (uint64_t k = 0; k < output_graph_degree; k++) {
        const uint64_t j = output_graph_ptr[k + (output_graph_degree * i)];
        if (j >= graph_size) continue;
#pragma omp atomic
        in_edge_count_ptr[j] += 1;
      }
    }
    auto hist     = raft::make_host_vector<uint32_t, int64_t>(output_graph_degree);
    auto hist_ptr = hist.data_handle();
    for (uint64_t k = 0; k < output_graph_degree; k++) {
      hist_ptr[k] = 0;
    }
#pragma omp parallel for
    for (uint64_t i = 0; i < graph_size; i++) {
      uint32_t count = in_edge_count_ptr[i];
      if (count >= output_graph_degree) continue;
#pragma omp atomic
      hist_ptr[count] += 1;
    }
    uint32_t sum_hist = 0;
    for (uint64_t k = 0; k < output_graph_degree; k++) {
      sum_hist += hist_ptr[k];
    }
  }
}

};  // end namespace Reachability
};  // end namespace detail
};  // end namespace HDBSCAN
};  // end namespace ML
