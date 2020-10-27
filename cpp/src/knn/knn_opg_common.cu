/*
* Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO LICENSEE:
*
* This source code and/or documentation ("Licensed Deliverables") are
* subject to NVIDIA intellectual property rights under U.S. and
* international Copyright laws.
*
* These Licensed Deliverables contained herein is PROPRIETARY and
* CONFIDENTIAL to NVIDIA and is being provided under the terms and
* conditions of a form of NVIDIA software license agreement by and
* between NVIDIA and Licensee ("License Agreement") or electronically
* accepted by Licensee.  Notwithstanding any terms or conditions to
* the contrary in the License Agreement, reproduction or disclosure
* of the Licensed Deliverables to any third party without the express
* written consent of NVIDIA is prohibited.
*
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
* SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
* PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
* NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
* DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
* NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
* SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
* DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
* WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
* ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
* OF THESE LICENSED DELIVERABLES.
*
* U.S. Government End Users.  These Licensed Deliverables are a
* "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
* 1995), consisting of "commercial computer software" and "commercial
* computer software documentation" as such terms are used in 48
* C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
* only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
* 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
* U.S. Government End Users acquire the Licensed Deliverables with
* only those rights set forth herein.
*
* Any use of the Licensed Deliverables in individual and commercial
* software must include, in the user documentation and internal
* comments to the code, the above Disclaimer and U.S. Government End
* Users Notice.
*/

#include <cuml/neighbors/knn_mg.hpp>
#include <selection/knn.cuh>

#include <common/cumlHandle.hpp>

#include <common/device_buffer.hpp>
#include <cuml/common/cuml_allocator.hpp>
#include <cuml/common/logger.hpp>
#include <raft/comms/comms.hpp>

#include <set>

#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>

namespace ML {
namespace KNN {
namespace opg {

namespace knn_common {

/**
 * This function copies the labels associated to the locally merged indices
 * from the index partitions to a merged array of labels
 * @param[out] out merged labels
 * @param[in] knn_indices merged indices
 * @param[in] parts unmerged labels in partitions
 * @param[in] offsets array splitting the partitions making it possible
 * to identify the origin partition of an nearest neighbor index
 * @param[in] cur_batch_size current batch size
 * @param[in] n_parts number of partitions
 * @param[in] n_labels number of labels to write (batch_size * n_outputs)
 */
template <typename T, int TPB_X>
__global__ void copy_label_outputs_from_index_parts_kernel(
  T *out, int64_t *knn_indices, T **parts, int64_t *offsets,
  size_t cur_batch_size, int n_parts, int n_labels) {
  int64_t i = (blockIdx.x * TPB_X) + threadIdx.x;
  if (i >= n_labels) return;
  int64_t nn_idx = knn_indices[i];
  int part_idx = 0;
  for (; part_idx < n_parts && nn_idx >= offsets[part_idx]; part_idx++)
    ;
  part_idx = min(max((int)0, part_idx - 1), n_parts - 1);
  int64_t offset = nn_idx - offsets[part_idx];
  out[i] = parts[part_idx][offset];
}

template <typename T>
void copy_label_outputs_from_index_parts(T *out, int64_t *knn_indices,
                                         std::vector<std::vector<T *>> &y,
                                         size_t cur_batch_size, int k,
                                         int n_outputs, int my_rank,
                                         Matrix::PartDescriptor &index_desc,
                                         std::shared_ptr<deviceAllocator> alloc,
                                         cudaStream_t stream) {
  const int TPB_X = 256;
  int n_labels = cur_batch_size * k;
  dim3 grid(raft::ceildiv(n_labels, TPB_X));
  dim3 blk(TPB_X);

  std::vector<Matrix::RankSizePair *> &idxPartsToRanks =
    index_desc.partsToRanks;
  int64_t offset = 0;
  std::vector<int64_t> offsets_h;
  for (auto &rsp : idxPartsToRanks) {
    if (rsp->rank == my_rank) {
      offsets_h.push_back(offset);
    }
    offset += rsp->size;
  }
  size_t n_parts = offsets_h.size();
  device_buffer<int64_t> offsets_d(alloc, stream, n_parts);
  raft::update_device(offsets_d.data(), offsets_h.data(), n_parts, stream);

  std::vector<T *> parts_h(n_parts);
  device_buffer<T *> parts_d(alloc, stream, n_parts);
  for (int o = 0; o < n_outputs; o++) {
    for (int p = 0; p < n_parts; p++) {
      parts_h[p] = y[p][o];
    }
    raft::update_device(parts_d.data(), parts_h.data(), n_parts, stream);

    copy_label_outputs_from_index_parts_kernel<T, TPB_X>
      <<<grid, blk, 0, stream>>>(out + (o * n_labels), knn_indices,
                                 parts_d.data(), offsets_d.data(),
                                 cur_batch_size, n_parts, n_labels);
  }
}

/**
 * This function copies the labels associated to the merged indices
 * from the unmerged to a merged (n_ranks times smaller) array of labels
 * @param[out] outputs merged labels
 * @param[in] knn_indices merged indices
 * @param[in] unmerged_outputs unmerged labels
 * @param[in] unmerged_knn_indices unmerged indices
 * @param[in] offsets array splitting the partitions making it possible
 * to identify the origin partition of an nearest neighbor index
 * @param[in] parts_to_ranks get rank index from index partition index,
 * informative to find positions as the unmerged arrays are built
 * so that ranks are in order (unlike partitions)
 * @param[in] nearest_neighbors number of nearest neighbors to look for in query
 * @param[in] n_outputs number of targets
 * @param[in] n_labels number of labels to write (batch_size * n_outputs)
 * @param[in] n_parts number of index partitions
 * @param[in] n_ranks number of index ranks
 */
template <typename T, int TPB_X>
__global__ void merge_labels_kernel(T *outputs, int64_t *knn_indices,
                                    T *unmerged_outputs,
                                    int64_t *unmerged_knn_indices,
                                    int64_t *offsets, int *parts_to_ranks,
                                    int nearest_neighbors, int n_outputs,
                                    int n_labels, int n_parts, int n_ranks) {
  int64_t i = (blockIdx.x * TPB_X) + threadIdx.x;
  if (i >= n_labels) return;
  int64_t nn_idx = knn_indices[i];
  int part_idx = 0;
  for (; part_idx < n_parts && nn_idx >= offsets[part_idx]; part_idx++)
    ;
  part_idx = min(max((int)0, part_idx - 1), n_parts - 1);
  int rank_idx = parts_to_ranks[part_idx];
  int inbatch_idx = i / nearest_neighbors;
  int64_t elm_idx = (rank_idx * n_labels) + inbatch_idx * nearest_neighbors;
  for (int k = 0; k < nearest_neighbors; k++) {
    if (nn_idx == unmerged_knn_indices[elm_idx + k]) {
      for (int o = 0; o < n_outputs; o++) {
        outputs[(o * n_labels) + i] =
          unmerged_outputs[(o * n_ranks * n_labels) + elm_idx + k];
      }
      return;
    }
  }
}

template <typename T>
void merge_labels(T *output, int64_t *knn_indices, T *unmerged_outputs,
                  int64_t *unmerged_knn_indices, int cur_batch_size,
                  int nearest_neighbors, int n_outputs,
                  Matrix::PartDescriptor &index_desc,
                  std::shared_ptr<deviceAllocator> alloc, cudaStream_t stream) {
  const int TPB_X = 256;
  int n_labels = cur_batch_size * nearest_neighbors;
  dim3 grid(raft::ceildiv(n_labels, TPB_X));
  dim3 blk(TPB_X);

  std::set<int> idxRanks = index_desc.uniqueRanks();
  std::vector<Matrix::RankSizePair *> &idxPartsToRanks =
    index_desc.partsToRanks;

  int offset = 0;
  std::vector<int64_t> offsets_h;
  for (auto &rsp : idxPartsToRanks) {
    offsets_h.push_back(offset);
    offset += rsp->size;
  }
  device_buffer<int64_t> offsets_d(alloc, stream, offsets_h.size());
  raft::update_device(offsets_d.data(), offsets_h.data(), offsets_h.size(),
                      stream);

  std::vector<int> parts_to_ranks_h;
  for (auto &rsp : idxPartsToRanks) {
    int i = 0;
    for (int rank : idxRanks) {
      if (rank == rsp->rank) {
        parts_to_ranks_h.push_back(i);
      }
      ++i;
    }
  }
  device_buffer<int> parts_to_ranks_d(alloc, stream, parts_to_ranks_h.size());
  raft::update_device(parts_to_ranks_d.data(), parts_to_ranks_h.data(),
                      parts_to_ranks_h.size(), stream);

  merge_labels_kernel<T, TPB_X><<<grid, blk, 0, stream>>>(
    output, knn_indices, unmerged_outputs, unmerged_knn_indices,
    offsets_d.data(), parts_to_ranks_d.data(), nearest_neighbors, n_outputs,
    n_labels, idxPartsToRanks.size(), idxRanks.size());
}

template <typename T>
void launch_local_operation(T *out, int64_t *knn_indices, std::vector<T *> y,
                            size_t total_labels, size_t cur_batch_size, int k,
                            const std::shared_ptr<deviceAllocator> alloc,
                            cudaStream_t stream, cudaStream_t *int_streams,
                            int n_int_streams, bool probas_only,
                            std::vector<float *> *probas,
                            std::vector<int *> *uniq_labels,
                            std::vector<int> *n_unique);

template <>
void launch_local_operation<int>(
  int *out, int64_t *knn_indices, std::vector<int *> y, size_t n_index_rows,
  size_t n_query_rows, int k, const std::shared_ptr<deviceAllocator> alloc,
  cudaStream_t stream, cudaStream_t *int_streams, int n_int_streams,
  bool probas_only, std::vector<float *> *probas,
  std::vector<int *> *uniq_labels, std::vector<int> *n_unique) {
  if (probas_only) {
    MLCommon::Selection::class_probs<32, true>(
      *probas, nullptr, y, n_index_rows, n_query_rows, k, *uniq_labels,
      *n_unique, alloc, stream, &int_streams[0], n_int_streams);
  } else {
    MLCommon::Selection::knn_classify<32, true>(
      out, nullptr, y, n_index_rows, n_query_rows, k, *uniq_labels, *n_unique,
      alloc, stream, &int_streams[0], n_int_streams);
  }
}

template <>
void launch_local_operation<float>(
  float *out, int64_t *knn_indices, std::vector<float *> y, size_t n_index_rows,
  size_t n_query_rows, int k, const std::shared_ptr<deviceAllocator> alloc,
  cudaStream_t stream, cudaStream_t *int_streams, int n_int_streams,
  bool probas_only, std::vector<float *> *probas,
  std::vector<int *> *uniq_labels, std::vector<int> *n_unique) {
  MLCommon::Selection::knn_regress<float, 32, true>(
    out, nullptr, y, n_index_rows, n_query_rows, k, stream, &int_streams[0],
    n_int_streams);
}

template <typename T>
void perform_local_operation(T *out, int64_t *knn_indices, T *labels,
                             size_t cur_batch_size, int k, int n_outputs,
                             raft::handle_t &h, bool probas_only = false,
                             std::vector<float *> *probas = nullptr,
                             std::vector<int *> *uniq_labels = nullptr,
                             std::vector<int> *n_unique = nullptr) {
  size_t n_labels = cur_batch_size * k;

  std::vector<T *> y(n_outputs);
  for (int o = 0; o < n_outputs; o++) {
    y[o] = labels + (o * n_labels);
  }

  cudaStream_t stream = h.get_stream();
  const auto alloc = h.get_device_allocator();

  int n_int_streams = h.get_num_internal_streams();
  cudaStream_t int_streams[n_int_streams];
  for (int i = 0; i < n_int_streams; i++) {
    int_streams[i] = h.get_internal_stream(i);
  }

  launch_local_operation(out, knn_indices, y, n_labels, cur_batch_size, k,
                         alloc, stream, int_streams, n_int_streams, probas_only,
                         probas, uniq_labels, n_unique);
}

template <typename T>
void reduce(raft::handle_t &handle, std::vector<Matrix::Data<T> *> *out,
            std::vector<Matrix::Data<int64_t> *> *out_I,
            std::vector<Matrix::floatData_t *> *out_D, device_buffer<T> &res,
            device_buffer<int64_t> &res_I, device_buffer<float> &res_D,
            Matrix::PartDescriptor &index_desc, size_t cur_batch_size, int k,
            int n_outputs, int local_parts_completed, size_t total_n_processed,
            bool probas_only = false,
            std::vector<std::vector<float *>> *probas = nullptr,
            std::vector<int *> *uniq_labels = nullptr,
            std::vector<int> *n_unique = nullptr) {
  const raft::handle_t &h = handle;
  cudaStream_t stream = h.get_stream();
  const auto alloc = h.get_device_allocator();

  std::set<int> idxRanks = index_desc.uniqueRanks();
  device_buffer<int64_t> trans(alloc, stream, idxRanks.size());
  CUDA_CHECK(cudaMemsetAsync(trans.data(), 0, idxRanks.size() * sizeof(int64_t),
                             stream));

  size_t batch_offset = total_n_processed * k;

  T *outputs = nullptr;
  int64_t *indices = nullptr;
  float *distances = nullptr;

  device_buffer<int64_t> *indices_b;
  device_buffer<float> *distances_b;
  std::vector<float *> probas_with_offsets;

  if (probas_only) {
    indices_b = new device_buffer<int64_t>(alloc, stream, cur_batch_size * k);
    distances_b = new device_buffer<float>(alloc, stream, cur_batch_size * k);
    indices = indices_b->data();
    distances = distances_b->data();

    std::vector<float *> &probas_part = probas->at(local_parts_completed);
    for (int i = 0; i < n_outputs; i++) {
      float *ptr = probas_part[i];
      int n_unique_classes = n_unique->at(i);
      probas_with_offsets.push_back(ptr +
                                    (total_n_processed * n_unique_classes));
    }
  } else {
    outputs =
      out->at(local_parts_completed)->ptr + (n_outputs * total_n_processed);
    indices = out_I->at(local_parts_completed)->ptr + batch_offset;
    distances = out_D->at(local_parts_completed)->ptr + batch_offset;
  }

  MLCommon::Selection::knn_merge_parts(res_D.data(), res_I.data(), distances,
                                       indices, cur_batch_size, idxRanks.size(),
                                       k, stream, trans.data());

  device_buffer<T> merged_outputs_b(alloc, stream,
                                    n_outputs * cur_batch_size * k);
  T *merged_outputs = merged_outputs_b.data();
  merge_labels(merged_outputs, indices, res.data(), res_I.data(),
               cur_batch_size, k, n_outputs, index_desc, alloc, stream);

  perform_local_operation<T>(outputs, indices, merged_outputs, cur_batch_size,
                             k, n_outputs, handle, probas_only,
                             &probas_with_offsets, uniq_labels, n_unique);

  if (probas_only) {
    delete indices_b;
    delete distances_b;
  }
}

void perform_local_knn(int64_t *res_I, float *res_D,
                       std::vector<Matrix::floatData_t *> &idx_data,
                       Matrix::PartDescriptor &idx_desc,
                       std::vector<Matrix::RankSizePair *> &local_idx_parts,
                       std::vector<size_t> &start_indices, cudaStream_t stream,
                       cudaStream_t *internal_streams, int n_internal_streams,
                       std::shared_ptr<deviceAllocator> allocator,
                       size_t cur_batch_size, int k, float *cur_query_ptr,
                       bool rowMajorIndex, bool rowMajorQuery) {
  std::vector<float *> ptrs(idx_data.size());
  std::vector<int> sizes(idx_data.size());

  for (int cur_idx = 0; cur_idx < idx_data.size(); cur_idx++) {
    ptrs[cur_idx] = idx_data[cur_idx]->ptr;
    sizes[cur_idx] = local_idx_parts[cur_idx]->size;
  }

  // PartDescriptor uses size_t while FAISS uses int64_t
  // so we need to do a quick conversion.
  std::vector<int64_t> start_indices_long;
  for (size_t start_index : start_indices)
    start_indices_long.push_back((int64_t)start_index);

  // ID ranges need to be offset by each local partition's
  // starting indices.
  MLCommon::Selection::brute_force_knn(
    ptrs, sizes, (int)idx_desc.N, cur_query_ptr, (int)cur_batch_size, res_I,
    res_D, k, allocator, stream, internal_streams, n_internal_streams,
    rowMajorIndex, rowMajorQuery, &start_indices_long);
}

void broadcast_query(float *query, size_t batch_input_elms, int part_rank,
                     std::set<int> idxRanks, const raft::comms::comms_t &comm,
                     cudaStream_t stream) {
  int my_rank = comm.get_rank();

  int request_idx = 0;
  std::vector<raft::comms::request_t> requests;
  if (part_rank == my_rank) {
    int idx_rank_size = idxRanks.size();
    if (idxRanks.find(my_rank) != idxRanks.end()) {
      --idx_rank_size;
    }

    requests.resize(idx_rank_size);

    for (int rank : idxRanks) {
      if (rank != my_rank) {
        comm.isend(query, batch_input_elms, rank, 0,
                   requests.data() + request_idx);
        ++request_idx;
      }
    }

  } else {
    requests.resize(1);
    comm.irecv(query, batch_input_elms, part_rank, 0,
               requests.data() + request_idx);
    ++request_idx;
  }

  try {
    comm.waitall(requests.size(), requests.data());
  } catch (raft::exception &e) {
    CUML_LOG_DEBUG("FAILURE!");
  }
}

/**
 * All non-root index ranks send the results for the current
 * query batch to the root rank for the batch.
 */
template <typename T>
void exchange_results(device_buffer<T> &res, device_buffer<int64_t> &res_I,
                      device_buffer<float> &res_D,
                      const raft::comms::comms_t &comm, int part_rank,
                      std::set<int> idxRanks, cudaStream_t stream,
                      std::shared_ptr<deviceAllocator> alloc,
                      size_t cur_batch_size, int k, int n_outputs,
                      int local_parts_completed) {
  int my_rank = comm.get_rank();

  size_t batch_elms = cur_batch_size * k;

  int request_idx = 0;
  std::vector<raft::comms::request_t> requests;
  if (part_rank != my_rank) {
    requests.resize(2 + n_outputs);
    comm.isend(res_I.data(), batch_elms, part_rank, 0,
               requests.data() + request_idx);
    ++request_idx;

    comm.isend(res_D.data(), batch_elms, part_rank, 0,
               requests.data() + request_idx);
    ++request_idx;

    for (size_t o = 0; o < n_outputs; o++) {
      comm.isend(res.data() + (o * batch_elms), batch_elms, part_rank, 0,
                 requests.data() + request_idx);
      ++request_idx;
    }
  } else {
    bool part_rank_is_idx = idxRanks.find(part_rank) != idxRanks.end();
    size_t idx_rank_size = idxRanks.size();

    // if root rank is an index, it will already have
    // query data, so no need to receive from it.
    res.resize(batch_elms * n_outputs * idx_rank_size, stream);
    res_I.resize(batch_elms * idx_rank_size, stream);
    res_D.resize(batch_elms * idx_rank_size, stream);
    if (part_rank_is_idx) {
      --idx_rank_size;
      int i = 0;
      for (int rank : idxRanks) {
        if (rank == my_rank) {
          size_t batch_offset = batch_elms * i;

          // Indices and distances are stored in rank order
          raft::copy_async(res_I.data() + batch_offset, res_I.data(),
                           batch_elms, stream);
          raft::copy_async(res_D.data() + batch_offset, res_D.data(),
                           batch_elms, stream);

          device_buffer<T> tmp_res(alloc, stream, n_outputs * batch_elms);
          raft::copy_async(tmp_res.data(), res.data(), tmp_res.size(), stream);

          for (int o = 0; o < n_outputs; ++o) {
            // Outputs are stored in target order and then in rank order
            raft::copy_async(
              res.data() + (o * idxRanks.size() * batch_elms) + batch_offset,
              tmp_res.data() + (o * batch_elms), batch_elms, stream);
          }
          CUDA_CHECK(cudaStreamSynchronize(stream));
          break;
        }
        i++;
      }
    }

    int num_received = 0;
    requests.resize((2 + n_outputs) * idx_rank_size);
    for (int rank : idxRanks) {
      if (rank != my_rank) {
        size_t batch_offset = batch_elms * num_received;

        // Indices and distances are stored in rank order
        comm.irecv(res_I.data() + batch_offset, batch_elms, rank, 0,
                   requests.data() + request_idx);
        ++request_idx;
        comm.irecv(res_D.data() + batch_offset, batch_elms, rank, 0,
                   requests.data() + request_idx);
        ++request_idx;

        for (size_t o = 0; o < n_outputs; o++) {
          // Outputs are stored in target order and then in rank order
          T *r = res.data() + (o * idxRanks.size() * batch_elms) + batch_offset;
          comm.irecv(r, batch_elms, rank, 0, requests.data() + request_idx);
          ++request_idx;
        }
        ++num_received;
      } else if (part_rank_is_idx) {
        /**
         * Prevents overwriting data when the owner of currently
         * processed query partition has itself some index partition(s)
         */
        ++num_received;
      }
    }
  }

  try {
    comm.waitall(requests.size(), requests.data());
  } catch (raft::exception &e) {
    CUML_LOG_DEBUG("FAILURE!");
  }
}

template <typename T>
void opg_knn(raft::handle_t &handle, std::vector<Matrix::Data<T> *> *out,
             std::vector<Matrix::Data<int64_t> *> *out_I,
             std::vector<Matrix::floatData_t *> *out_D,
             std::vector<Matrix::floatData_t *> &idx_data,
             Matrix::PartDescriptor &idx_desc,
             std::vector<Matrix::floatData_t *> &query_data,
             Matrix::PartDescriptor &query_desc,
             std::vector<std::vector<T *>> &y, bool rowMajorIndex,
             bool rowMajorQuery, int k, int n_outputs, size_t batch_size,
             bool verbose, std::vector<std::vector<float *>> *probas = nullptr,
             std::vector<int *> *uniq_labels = nullptr,
             std::vector<int> *n_unique = nullptr, bool probas_only = false) {
  ASSERT(k <= 1024, "k must be <= 1024");
  ASSERT(batch_size > 0, "max_batch_size must be > 0");
  ASSERT(k < idx_desc.M, "k must be less than the total number of query rows");
  for (Matrix::RankSizePair *rsp : idx_desc.partsToRanks) {
    ASSERT(rsp->size >= k,
           "k must be <= the number of rows in the smallest index partition.");
  }

  const raft::handle_t &h = handle;
  const auto &comm = h.get_comms();
  cudaStream_t stream = h.get_stream();

  const auto allocator = h.get_device_allocator();

  int my_rank = comm.get_rank();

  std::set<int> idxRanks = idx_desc.uniqueRanks();

  std::vector<Matrix::RankSizePair *> local_idx_parts =
    idx_desc.blocksOwnedBy(comm.get_rank());

  int local_parts_completed = 0;

  // Loop through query parts for all ranks
  for (int i = 0; i < query_desc.totalBlocks(); i++) {
    Matrix::RankSizePair *partition = query_desc.partsToRanks[i];
    int part_rank = partition->rank;
    size_t part_n_rows = partition->size;

    size_t total_batches = raft::ceildiv(part_n_rows, batch_size);
    size_t total_n_processed = 0;

    // Loop through batches for each query part
    for (int cur_batch = 0; cur_batch < total_batches; cur_batch++) {
      size_t cur_batch_size = batch_size;

      if (cur_batch == total_batches - 1)
        cur_batch_size = part_n_rows - (cur_batch * batch_size);

      if (my_rank == part_rank) CUML_LOG_DEBUG("Root Rank is %d", my_rank);

      /**
       * Root broadcasts batch to all other ranks
       */
      CUML_LOG_DEBUG("Rank %d: Performing Broadcast", my_rank);

      int my_rank = comm.get_rank();
      device_buffer<float> part_data(allocator, stream, 0);

      size_t batch_input_elms = cur_batch_size * query_desc.N;
      size_t batch_input_offset = batch_input_elms * cur_batch;

      float *cur_query_ptr;

      device_buffer<float> tmp_batch_buf(allocator, stream, 0);
      // current partition's owner rank broadcasts
      if (part_rank == my_rank) {
        Matrix::Data<float> *data = query_data[local_parts_completed];

        // If query is column major and total_batches > 0, create a
        // temporary buffer for the batch so that we can stack rows.
        if (!rowMajorQuery && total_batches > 1) {
          tmp_batch_buf.resize(batch_input_elms, stream);
          for (int col_data = 0; col_data < query_desc.N; col_data++) {
            raft::copy(
              tmp_batch_buf.data() + (col_data * cur_batch_size),
              data->ptr + ((col_data * part_n_rows) + total_n_processed),
              cur_batch_size, stream);
          }
          cur_query_ptr = tmp_batch_buf.data();

        } else {
          cur_query_ptr = data->ptr + batch_input_offset;
        }

        // all other (index) ranks receive
      } else if (idxRanks.find(my_rank) != idxRanks.end()) {
        part_data.resize(batch_input_elms, stream);
        cur_query_ptr = part_data.data();
      }

      bool my_rank_is_idx = idxRanks.find(my_rank) != idxRanks.end();

      /**
       * Send query to index partitions
       */
      if (my_rank == part_rank || my_rank_is_idx)
        broadcast_query(cur_query_ptr, batch_input_elms, part_rank, idxRanks,
                        comm, stream);

      device_buffer<T> res(allocator, stream);
      device_buffer<int64_t> res_I(allocator, stream);
      device_buffer<float> res_D(allocator, stream);
      if (my_rank_is_idx) {
        /**
         * All index ranks perform local KNN
         */
        CUML_LOG_DEBUG("Rank %d: Performing Local KNN", my_rank);

        size_t batch_knn_elms = k * cur_batch_size;

        res.resize(batch_knn_elms * n_outputs, stream);
        res_I.resize(batch_knn_elms, stream);
        res_D.resize(batch_knn_elms, stream);

        // Offset nearest neighbor index matrix by partition indices
        std::vector<size_t> start_indices = idx_desc.startIndices(my_rank);

        cudaStream_t int_streams[handle.get_num_internal_streams()];
        for (int i = 0; i < handle.get_num_internal_streams(); i++) {
          int_streams[i] = handle.get_internal_stream(i);
        }

        perform_local_knn(res_I.data(), res_D.data(), idx_data, idx_desc,
                          local_idx_parts, start_indices, stream, int_streams,
                          handle.get_num_internal_streams(),
                          handle.get_device_allocator(), cur_batch_size, k,
                          cur_query_ptr, rowMajorIndex, rowMajorQuery);

        copy_label_outputs_from_index_parts(
          res.data(), res_I.data(), y, (size_t)cur_batch_size, (int)k,
          (int)n_outputs, my_rank, idx_desc, handle.get_device_allocator(),
          stream);

        // Synchronize before sending
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaPeekAtLastError());
      }

      if (part_rank == my_rank || my_rank_is_idx) {
        /**
         * Ranks exchange results.
         * Each rank having index partition(s) sends
         * its local results (my_rank_is_idx)
         * Additionally the owner of currently processed query partition
         * receives and performs a reduce even if it has
         * no index partition (part_rank == my_rank)
         */
        CUML_LOG_DEBUG("Rank %d: Exchanging results", my_rank);
        exchange_results(res, res_I, res_D, comm, part_rank, idxRanks, stream,
                         handle.get_device_allocator(), cur_batch_size, k,
                         n_outputs, local_parts_completed);
      }

      /**
       * Root rank performs local reduce
       */
      if (part_rank == my_rank) {
        CUML_LOG_DEBUG("Rank %d: Performing Reduce", my_rank);

        reduce(handle, out, out_I, out_D, res, res_I, res_D, idx_desc,
               cur_batch_size, k, n_outputs, local_parts_completed,
               total_n_processed, probas_only, probas, uniq_labels, n_unique);

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaPeekAtLastError());

        CUML_LOG_DEBUG("Rank %d: Finished Reduce", my_rank);
      }

      total_n_processed += cur_batch_size;
    }

    if (my_rank == part_rank) local_parts_completed++;
  }
};

template void opg_knn<int>(raft::handle_t &handle,
                           std::vector<Matrix::Data<int> *> *out,
                           std::vector<Matrix::Data<int64_t> *> *out_I,
                           std::vector<Matrix::floatData_t *> *out_D,
                           std::vector<Matrix::floatData_t *> &idx_data,
                           Matrix::PartDescriptor &idx_desc,
                           std::vector<Matrix::floatData_t *> &query_data,
                           Matrix::PartDescriptor &query_desc,
                           std::vector<std::vector<int *>> &y,
                           bool rowMajorIndex, bool rowMajorQuery, int k,
                           int n_outputs, size_t batch_size, bool verbose,
                           std::vector<std::vector<float *>> *probas,
                           std::vector<int *> *uniq_labels,
                           std::vector<int> *n_unique, bool probas_only);

template void opg_knn<float>(raft::handle_t &handle,
                             std::vector<Matrix::Data<float> *> *out,
                             std::vector<Matrix::Data<int64_t> *> *out_I,
                             std::vector<Matrix::floatData_t *> *out_D,
                             std::vector<Matrix::floatData_t *> &idx_data,
                             Matrix::PartDescriptor &idx_desc,
                             std::vector<Matrix::floatData_t *> &query_data,
                             Matrix::PartDescriptor &query_desc,
                             std::vector<std::vector<float *>> &y,
                             bool rowMajorIndex, bool rowMajorQuery, int k,
                             int n_outputs, size_t batch_size, bool verbose,
                             std::vector<std::vector<float *>> *probas,
                             std::vector<int *> *uniq_labels,
                             std::vector<int> *n_unique, bool probas_only);

};  // namespace knn_common
};  // namespace opg
};  // namespace KNN
};  // namespace ML
