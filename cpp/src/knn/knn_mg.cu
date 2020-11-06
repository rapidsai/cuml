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

#include <cuml/neighbors/knn_mg.hpp>
#include <selection/knn.cuh>

#include <common/cumlHandle.hpp>

#include <common/device_buffer.hpp>
#include <cuml/common/cuml_allocator.hpp>
#include <raft/comms/comms.hpp>

#include <set>

#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>

namespace ML {
namespace KNN {
namespace opg {

void reduce(Matrix::Data<int64_t> *&out_I, Matrix::floatData_t *&out_D,
            device_buffer<int64_t> &res_I, device_buffer<float> &res_D,
            Matrix::PartDescriptor &index_desc,
            const raft::comms::comms_t &comm,
            std::shared_ptr<deviceAllocator> alloc, cudaStream_t stream,
            size_t cur_batch_size, int k, int local_parts_completed,
            int cur_batch, size_t total_n_processed, std::set<int> idxRanks) {
  Matrix::Data<int64_t> *I = out_I;
  Matrix::floatData_t *D = out_D;

  size_t batch_offset = total_n_processed * k;

  device_buffer<int64_t> trans(alloc, stream, idxRanks.size());
  CUDA_CHECK(cudaMemsetAsync(trans.data(), 0, idxRanks.size() * sizeof(int64_t),
                             stream));

  MLCommon::Selection::knn_merge_parts(
    res_D.data(), res_I.data(), D->ptr + batch_offset, I->ptr + batch_offset,
    cur_batch_size, idxRanks.size(), k, stream, trans.data());
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
    std::cout << "FAILRE!" << std::endl;
  }
}

/**
   * All non-root index ranks send the results for the current
   * query batch to the root rank for the batch.
   */
void exchange_results(device_buffer<int64_t> &res_I,
                      device_buffer<float> &res_D,
                      const raft::comms::comms_t &comm, int part_rank,
                      std::set<int> idxRanks, cudaStream_t stream,
                      size_t cur_batch_size, int k, int local_parts_completed) {
  int my_rank = comm.get_rank();

  size_t batch_elms = cur_batch_size * k;

  int request_idx = 0;
  std::vector<raft::comms::request_t> requests;
  if (part_rank != my_rank) {
    requests.resize(2);
    comm.isend(res_I.data(), batch_elms, part_rank, 0,
               requests.data() + request_idx);
    ++request_idx;

    comm.isend(res_D.data(), batch_elms, part_rank, 0,
               requests.data() + request_idx);
    ++request_idx;

  } else {
    bool part_rank_is_idx = idxRanks.find(part_rank) != idxRanks.end();
    int idx_rank_size = idxRanks.size();

    int num_received = 0;

    // if root rank is an index, it will already have
    // query data, so no need to receive from it.
    if (part_rank_is_idx) {
      num_received = 1;  // root rank will take the zeroth slot
      res_I.resize(batch_elms * idx_rank_size, stream);
      res_D.resize(batch_elms * idx_rank_size, stream);
      --idx_rank_size;
    } else {
      res_I.resize(batch_elms * idx_rank_size, stream);
      res_D.resize(batch_elms * idx_rank_size, stream);
    }

    requests.resize(2 * idx_rank_size);
    for (int rank : idxRanks) {
      if (rank != my_rank) {
        size_t batch_offset = batch_elms * num_received;

        comm.irecv(res_I.data() + batch_offset, batch_elms, rank, 0,
                   requests.data() + request_idx);
        ++request_idx;
        comm.irecv(res_D.data() + batch_offset, batch_elms, rank, 0,
                   requests.data() + request_idx);
        ++request_idx;
        ++num_received;
      }
    }
  }

  try {
    comm.waitall(requests.size(), requests.data());
  } catch (raft::exception &e) {
    std::cout << "FAILURE!" << std::endl;
  }
}

void brute_force_knn(raft::handle_t &handle,
                     std::vector<Matrix::Data<int64_t> *> &out_I,
                     std::vector<Matrix::floatData_t *> &out_D,
                     std::vector<Matrix::floatData_t *> &idx_data,
                     Matrix::PartDescriptor &idx_desc,
                     std::vector<Matrix::floatData_t *> &query_data,
                     Matrix::PartDescriptor &query_desc, bool rowMajorIndex,
                     bool rowMajorQuery, int k, size_t batch_size,
                     bool verbose) {
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

      if (my_rank == part_rank && verbose) {
        std::cout << "Root Rank is " << my_rank << std::endl;
      }

      /**
         * Root broadcasts batch to all other ranks
         */
      if (verbose) {
        std::cout << "Rank " << my_rank << ": Performing Broadcast"
                  << std::endl;
      }

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

      device_buffer<int64_t> res_I(allocator, stream);
      device_buffer<float> res_D(allocator, stream);
      if (my_rank_is_idx) {
        /**
           * All index ranks perform local KNN
           */
        if (verbose)
          std::cout << "Rank " << my_rank << ": Performing Local KNN"
                    << std::endl;

        size_t batch_knn_elms = k * cur_batch_size;

        res_I.resize(batch_knn_elms, stream);
        res_D.resize(batch_knn_elms, stream);

        // Offset nearest neighbor index matrix by partition indices
        std::vector<size_t> start_indices = idx_desc.startIndices(my_rank);

        cudaStream_t int_streams[handle.get_num_internal_streams()];
        for (int i = 0; i < handle.get_num_internal_streams(); i++) {
          int_streams[i] = handle.get_internal_stream(i);
        }

        perform_local_knn(res_I.data(), res_D.data(), idx_data, idx_desc,
                          local_idx_parts, start_indices, stream, &*int_streams,
                          handle.get_num_internal_streams(),
                          handle.get_device_allocator(), cur_batch_size, k,
                          cur_query_ptr, rowMajorIndex, rowMajorQuery);

        // Synchronize before sending
        CUDA_CHECK(cudaStreamSynchronize(stream));

        /**
           * Ranks exchange results.
           * Partition owner receives. All other ranks send.
           */
        if (verbose)
          std::cout << "Rank " << my_rank << ": Exchanging results"
                    << std::endl;
        exchange_results(res_I, res_D, comm, part_rank, idxRanks, stream,
                         cur_batch_size, k, local_parts_completed);
      }

      /**
         * Root rank performs local reduce
         */
      if (part_rank == my_rank) {
        if (verbose)
          std::cout << "Rank " << my_rank << ": Performing Reduce" << std::endl;

        reduce(out_I[local_parts_completed], out_D[local_parts_completed],
               res_I, res_D, idx_desc, comm, allocator, stream, cur_batch_size,
               k, local_parts_completed, cur_batch, total_n_processed,
               idxRanks);

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaPeekAtLastError());

        if (verbose)
          std::cout << "Rank " << my_rank << ": Finished Reduce" << std::endl;
      }

      total_n_processed += cur_batch_size;
    }

    if (my_rank == part_rank) local_parts_completed++;
  }
}

};  // namespace opg
};  // namespace KNN
};  // namespace ML
