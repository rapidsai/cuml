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

#include "knn_opg_common.cuh"

namespace ML {
namespace KNN {
namespace opg {

namespace knn_common {

void opg_knn(opg_knn_param &params, cuda_utils &cutils) {
  opg_knn_utils utils(params, cutils);

  ASSERT(params.k <= 1024, "k must be <= 1024");
  ASSERT(params.batch_size > 0, "max_batch_size must be > 0");
  ASSERT(params.k < params.idx_desc->M,
         "k must be less than the total number of query rows");
  for (Matrix::RankSizePair *rsp : utils.idxPartsToRanks) {
    ASSERT(rsp->size >= params.k,
           "k must be <= the number of rows in the smallest index partition.");
  }

  int local_parts_completed = 0;
  // Loop through query parts for all ranks
  for (int i = 0; i < params.query_desc->totalBlocks();
       i++) {  // For each query partitions
    Matrix::RankSizePair *partition = utils.queryPartsToRanks[i];
    int part_rank = partition->rank;
    size_t part_n_rows = partition->size;

    size_t total_batches = raft::ceildiv(part_n_rows, params.batch_size);
    size_t total_n_processed = 0;

    // Loop through batches for each query part
    for (int cur_batch = 0; cur_batch < total_batches;
         cur_batch++) {  // For each batch in a query partition
      size_t cur_batch_size = params.batch_size;

      if (cur_batch == total_batches - 1)
        cur_batch_size = part_n_rows - (cur_batch * params.batch_size);

      if (utils.my_rank == part_rank)
        CUML_LOG_DEBUG("Root Rank is %d", utils.my_rank);

      /**
        * Root broadcasts batch to all other ranks
        */
      CUML_LOG_DEBUG("Rank %d: Performing Broadcast", utils.my_rank);

      device_buffer<float> part_data(cutils.alloc, cutils.stream, 0);

      size_t batch_input_elms = cur_batch_size * params.query_desc->N;
      size_t batch_input_offset = batch_input_elms * cur_batch;

      float *cur_query_ptr;

      device_buffer<float> tmp_batch_buf(cutils.alloc, cutils.stream, 0);
      // current partition's owner rank broadcasts
      if (part_rank == utils.my_rank) {
        Matrix::Data<float> *data =
          params.query_data->at(local_parts_completed);

        // If query is column major and total_batches > 0, create a
        // temporary buffer for the batch so that we can stack rows.
        if (!params.rowMajorQuery && total_batches > 1) {
          tmp_batch_buf.resize(batch_input_elms, cutils.stream);
          for (int col_data = 0; col_data < params.query_desc->N; col_data++) {
            raft::copy(
              tmp_batch_buf.data() + (col_data * cur_batch_size),
              data->ptr + ((col_data * part_n_rows) + total_n_processed),
              cur_batch_size, cutils.stream);
          }
          cur_query_ptr = tmp_batch_buf.data();

        } else {
          cur_query_ptr = data->ptr + batch_input_offset;
        }

        // all other (index) ranks receive
      } else if (utils.idxRanks.find(utils.my_rank) != utils.idxRanks.end()) {
        part_data.resize(batch_input_elms, cutils.stream);
        cur_query_ptr = part_data.data();
      }

      bool my_rank_is_idx =
        utils.idxRanks.find(utils.my_rank) != utils.idxRanks.end();

      /**
        * Send query to index partitions
        */
      if (utils.my_rank == part_rank || my_rank_is_idx)
        broadcast_query(utils, cutils, part_rank, cur_query_ptr,
                        batch_input_elms);

      if (my_rank_is_idx) {
        /**
          * All index ranks perform local KNN
          */
        CUML_LOG_DEBUG("Rank %d: Performing Local KNN", utils.my_rank);

        size_t batch_knn_elms = params.k * cur_batch_size;

        if (params.knn_op != knn_operation::knn) {
          // No labels for KNN only operation
          utils.res->resize(batch_knn_elms * params.n_outputs, cutils.stream);
        }
        utils.res_I->resize(batch_knn_elms, cutils.stream);
        utils.res_D->resize(batch_knn_elms, cutils.stream);

        // Perform a local KNN search
        perform_local_knn(params, utils, cutils, cur_query_ptr, cur_batch_size);
        CUDA_CHECK(cudaStreamSynchronize(cutils.stream));
        CUDA_CHECK(cudaPeekAtLastError());

        if (params.knn_op != knn_operation::knn) {
          // Get the right labels for indices obtained after a KNN merge
          copy_label_outputs_from_index_parts(params, utils, cutils,
                                              cur_batch_size);
          CUDA_CHECK(cudaStreamSynchronize(cutils.stream));
          CUDA_CHECK(cudaPeekAtLastError());
        }
      }

      if (part_rank == utils.my_rank || my_rank_is_idx) {
        /**
          * Ranks exchange results.
          * Each rank having index partition(s) sends
          * its local results (my_rank_is_idx)
          * Additionally the owner of currently processed query partition
          * receives and performs a reduce even if it has
          * no index partition (part_rank == my_rank)
          */
        CUML_LOG_DEBUG("Rank %d: Exchanging results", utils.my_rank);
        exchange_results(params, utils, cutils, part_rank, cur_batch_size);
      }

      /**
        * Root rank performs local reduce
        */
      if (part_rank == utils.my_rank) {
        CUML_LOG_DEBUG("Rank %d: Performing Reduce", utils.my_rank);

        // Reduce all local results to a global result for a given query batch
        reduce(params, utils, cutils, local_parts_completed, total_n_processed,
               cur_batch_size);
        CUDA_CHECK(cudaStreamSynchronize(cutils.stream));
        CUDA_CHECK(cudaPeekAtLastError());

        CUML_LOG_DEBUG("Rank %d: Finished Reduce", utils.my_rank);
      }

      total_n_processed += cur_batch_size;
    }

    if (utils.my_rank == part_rank) local_parts_completed++;
  }
};

void broadcast_query(opg_knn_utils &utils, cuda_utils &cutils, int part_rank,
                     float *broadcast, size_t broadcast_size) {
  int request_idx = 0;
  std::vector<raft::comms::request_t> requests;
  if (part_rank == utils.my_rank) {  // Either broadcast to other workers
    int idx_rank_size = utils.idxRanks.size();
    if (utils.idxRanks.find(utils.my_rank) != utils.idxRanks.end()) {
      --idx_rank_size;
    }

    requests.resize(idx_rank_size);

    for (int rank : utils.idxRanks) {
      if (rank != utils.my_rank) {
        cutils.comm->isend(broadcast, broadcast_size, rank, 0,
                           requests.data() + request_idx);
        ++request_idx;
      }
    }

  } else {  // Or receive from broadcaster
    requests.resize(1);
    cutils.comm->irecv(broadcast, broadcast_size, part_rank, 0,
                       requests.data() + request_idx);
    ++request_idx;
  }

  try {
    cutils.comm->waitall(requests.size(), requests.data());
  } catch (raft::exception &e) {
    CUML_LOG_DEBUG("FAILURE!");
  }
}

void perform_local_knn(opg_knn_param &params, opg_knn_utils &utils,
                       cuda_utils &cutils, float *query, size_t query_size) {
  std::vector<float *> ptrs(params.idx_data->size());
  std::vector<int> sizes(params.idx_data->size());

  for (int cur_idx = 0; cur_idx < params.idx_data->size(); cur_idx++) {
    ptrs[cur_idx] = params.idx_data->at(cur_idx)->ptr;
    sizes[cur_idx] = utils.local_idx_parts[cur_idx]->size;
  }

  // Offset nearest neighbor index matrix by partition indices
  std::vector<size_t> start_indices =
    params.idx_desc->startIndices(utils.my_rank);
  // PartDescriptor uses size_t while FAISS uses int64_t
  // so we need to do a quick conversion.
  std::vector<int64_t> start_indices_long;
  for (size_t start_index : start_indices)
    start_indices_long.push_back((int64_t)start_index);

  // ID ranges need to be offset by each local partition's
  // starting indices.
  MLCommon::Selection::brute_force_knn(
    ptrs, sizes, params.idx_desc->N, query, query_size, utils.res_I->data(),
    utils.res_D->data(), params.k, cutils.alloc, cutils.stream,
    cutils.internal_streams.data(), cutils.internal_streams.size(),
    params.rowMajorIndex, params.rowMajorQuery, &start_indices_long);
  CUDA_CHECK(cudaStreamSynchronize(cutils.stream));
  CUDA_CHECK(cudaPeekAtLastError());
}

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
template <int TPB_X>
__global__ void copy_label_outputs_from_index_parts_kernel(
  char32_t *out, int64_t *knn_indices, char32_t **parts, int64_t *offsets,
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

void copy_label_outputs_from_index_parts(opg_knn_param &params,
                                         opg_knn_utils &utils,
                                         cuda_utils &cutils,
                                         size_t batch_size) {
  const int TPB_X = 256;
  int n_labels = batch_size * params.k;
  dim3 grid(raft::ceildiv(n_labels, TPB_X));
  dim3 blk(TPB_X);

  int64_t offset = 0;
  std::vector<int64_t> offsets_h;
  for (auto &rsp : utils.idxPartsToRanks) {
    if (rsp->rank == utils.my_rank) {
      offsets_h.push_back(offset);
    }
    offset += rsp->size;
  }
  size_t n_parts = offsets_h.size();
  device_buffer<int64_t> offsets_d(cutils.alloc, cutils.stream, n_parts);
  raft::update_device(offsets_d.data(), offsets_h.data(), n_parts,
                      cutils.stream);

  std::vector<char32_t *> parts_h(n_parts);
  device_buffer<char32_t *> parts_d(cutils.alloc, cutils.stream, n_parts);
  for (int o = 0; o < params.n_outputs; o++) {
    if (params.knn_op == knn_operation::regression) {
      for (int p = 0; p < n_parts; p++) {
        parts_h[p] = (char32_t *)(params.y.f->at(p)[o]);
      }
    } else {
      for (int p = 0; p < n_parts; p++) {
        parts_h[p] = (char32_t *)(params.y.i->at(p)[o]);
      }
    }
    raft::update_device(parts_d.data(), parts_h.data(), n_parts, cutils.stream);

    copy_label_outputs_from_index_parts_kernel<TPB_X>
      <<<grid, blk, 0, cutils.stream>>>(
        utils.res->data() + (o * n_labels), utils.res_I->data(), parts_d.data(),
        offsets_d.data(), batch_size, n_parts, n_labels);
  }
}

/**
 * All non-root index ranks send the results for the current
 * query batch to the root rank for the batch.
 */
void exchange_results(opg_knn_param &params, opg_knn_utils &utils,
                      cuda_utils &cutils, int part_rank, size_t batch_size) {
  size_t batch_elms = batch_size * params.k;

  int request_idx = 0;
  std::vector<raft::comms::request_t> requests;
  if (part_rank != utils.my_rank) {  // Either send local KNN results
    requests.resize(2);
    cutils.comm->isend(utils.res_I->data(), batch_elms, part_rank, 0,
                       requests.data() + request_idx);
    ++request_idx;

    cutils.comm->isend(utils.res_D->data(), batch_elms, part_rank, 0,
                       requests.data() + request_idx);
    ++request_idx;

    if (params.knn_op != knn_operation::knn) {
      requests.resize(2 + params.n_outputs);
      for (size_t o = 0; o < params.n_outputs; o++) {
        cutils.comm->isend(utils.res->data() + (o * batch_elms), batch_elms,
                           part_rank, 0, requests.data() + request_idx);
        ++request_idx;
      }
    }
  } else {  // Or, as the owner of currently processed query batch,
            // receive results from other workers for reduce
    bool part_rank_is_idx =
      utils.idxRanks.find(part_rank) != utils.idxRanks.end();
    size_t idx_rank_size = utils.idxRanks.size();

    // if root rank is an index, it will already have
    // query data, so no need to receive from it.
    utils.res_I->resize(batch_elms * idx_rank_size, cutils.stream);
    utils.res_D->resize(batch_elms * idx_rank_size, cutils.stream);

    if (params.knn_op != knn_operation::knn) {
      utils.res->resize(batch_elms * params.n_outputs * idx_rank_size,
                        cutils.stream);
    }

    if (part_rank_is_idx) {
      /**
       * If this worker (in charge of reduce),
       * has some local results as well,
       * copy them at right location
       */
      --idx_rank_size;
      int i = 0;
      for (int rank : utils.idxRanks) {
        if (rank == utils.my_rank) {
          size_t batch_offset = batch_elms * i;

          // Indices and distances are stored in rank order
          raft::copy_async(utils.res_I->data() + batch_offset,
                           utils.res_I->data(), batch_elms, cutils.stream);
          raft::copy_async(utils.res_D->data() + batch_offset,
                           utils.res_D->data(), batch_elms, cutils.stream);

          if (params.knn_op != knn_operation::knn) {
            device_buffer<char32_t> tmp_res(cutils.alloc, cutils.stream,
                                            params.n_outputs * batch_elms);
            raft::copy_async(tmp_res.data(), utils.res->data(), tmp_res.size(),
                             cutils.stream);

            for (int o = 0; o < params.n_outputs; ++o) {
              // Outputs are stored in target order and then in rank order
              raft::copy_async(
                utils.res->data() + (o * utils.idxRanks.size() * batch_elms) +
                  batch_offset,
                tmp_res.data() + (o * batch_elms), batch_elms, cutils.stream);
            }
          }
          CUDA_CHECK(cudaStreamSynchronize(cutils.stream));
          break;
        }
        i++;
      }
    }

    size_t request_size = 2 * idx_rank_size;
    if (params.knn_op != knn_operation::knn)
      request_size = (2 + params.n_outputs) * idx_rank_size;
    requests.resize(request_size);

    int num_received = 0;
    for (int rank : utils.idxRanks) {
      if (rank != utils.my_rank) {
        size_t batch_offset = batch_elms * num_received;

        // Indices and distances are stored in rank order
        cutils.comm->irecv(utils.res_I->data() + batch_offset, batch_elms, rank,
                           0, requests.data() + request_idx);
        ++request_idx;
        cutils.comm->irecv(utils.res_D->data() + batch_offset, batch_elms, rank,
                           0, requests.data() + request_idx);
        ++request_idx;

        if (params.knn_op != knn_operation::knn) {
          for (size_t o = 0; o < params.n_outputs; o++) {
            // Outputs are stored in target order and then in rank order
            char32_t *r = utils.res->data() +
                          (o * utils.idxRanks.size() * batch_elms) +
                          batch_offset;
            cutils.comm->irecv(r, batch_elms, rank, 0,
                               requests.data() + request_idx);
            ++request_idx;
          }
        }
      }
      if (rank != utils.my_rank || part_rank_is_idx) {
        /**
          * Increase index for each new reception
          * Also increase index when the worker doing a reduce operation
          * has some index data (previously copied at right location).
          */
        ++num_received;
      }
    }
  }

  try {
    cutils.comm->waitall(requests.size(), requests.data());
  } catch (raft::exception &e) {
    CUML_LOG_DEBUG("FAILURE!");
  }
}

void reduce(opg_knn_param &params, opg_knn_utils &utils, cuda_utils &cutils,
            int part_idx, size_t processed_in_part, size_t batch_size) {
  device_buffer<int64_t> trans(cutils.alloc, cutils.stream,
                               utils.idxRanks.size());
  CUDA_CHECK(cudaMemsetAsync(
    trans.data(), 0, utils.idxRanks.size() * sizeof(int64_t), cutils.stream));

  size_t batch_offset = processed_in_part * params.k;

  char32_t *outputs = nullptr;
  int64_t *indices = nullptr;
  float *distances = nullptr;

  device_buffer<int64_t> *indices_b;
  device_buffer<float> *distances_b;
  std::vector<float *> probas_with_offsets;

  if (params.knn_op == knn_operation::knn) {
    indices = params.out_I->at(part_idx)->ptr + batch_offset;
    distances = params.out_D->at(part_idx)->ptr + batch_offset;
  } else {
    indices_b = new device_buffer<int64_t>(cutils.alloc, cutils.stream,
                                           batch_size * params.k);
    distances_b = new device_buffer<float>(cutils.alloc, cutils.stream,
                                           batch_size * params.k);
    indices = indices_b->data();
    distances = distances_b->data();
  }

  if (params.knn_op == knn_operation::class_proba) {
    std::vector<float *> &probas_part = params.probas->at(part_idx);
    for (int i = 0; i < params.n_outputs; i++) {
      float *ptr = probas_part[i];
      int n_unique_classes = params.n_unique->at(i);
      probas_with_offsets.push_back(ptr +
                                    (processed_in_part * n_unique_classes));
    }
  } else {
    if (params.knn_op == knn_operation::classification) {
      outputs = (char32_t *)params.out.i->at(part_idx)->ptr +
                (params.n_outputs * processed_in_part);
    } else if (params.knn_op == knn_operation::regression) {
      outputs = (char32_t *)params.out.f->at(part_idx)->ptr +
                (params.n_outputs * processed_in_part);
    }
  }

  // Merge all KNN local results
  MLCommon::Selection::knn_merge_parts(
    utils.res_D->data(), utils.res_I->data(), distances, indices, batch_size,
    utils.idxRanks.size(), params.k, cutils.stream, trans.data());
  CUDA_CHECK(cudaStreamSynchronize(cutils.stream));
  CUDA_CHECK(cudaPeekAtLastError());

  if (params.knn_op != knn_operation::knn) {
    device_buffer<char32_t> merged_outputs_b(
      cutils.alloc, cutils.stream, params.n_outputs * batch_size * params.k);
    // Get the right labels for indices obtained after local KNN searches
    merge_labels(params, utils, cutils, merged_outputs_b.data(), indices,
                 utils.res->data(), utils.res_I->data(), batch_size);
    CUDA_CHECK(cudaStreamSynchronize(cutils.stream));
    CUDA_CHECK(cudaPeekAtLastError());

    // Perform final classification, regression or class-proba operation
    perform_local_operation(params, utils, cutils, outputs, probas_with_offsets,
                            merged_outputs_b.data(), batch_size);
    CUDA_CHECK(cudaStreamSynchronize(cutils.stream));
    CUDA_CHECK(cudaPeekAtLastError());
  }

  if (params.knn_op == knn_operation::class_proba) {
    delete indices_b;
    delete distances_b;
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
template <int TPB_X>
__global__ void merge_labels_kernel(char32_t *outputs, int64_t *knn_indices,
                                    char32_t *unmerged_outputs,
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

void merge_labels(opg_knn_param &params, opg_knn_utils &utils,
                  cuda_utils &cutils, char32_t *output, int64_t *knn_indices,
                  char32_t *unmerged_outputs, int64_t *unmerged_knn_indices,
                  int batch_size) {
  const int TPB_X = 256;
  int n_labels = batch_size * params.k;
  dim3 grid(raft::ceildiv(n_labels, TPB_X));
  dim3 blk(TPB_X);

  int offset = 0;
  std::vector<int64_t> offsets_h;
  for (auto &rsp : utils.idxPartsToRanks) {
    offsets_h.push_back(offset);
    offset += rsp->size;
  }
  device_buffer<int64_t> offsets_d(cutils.alloc, cutils.stream,
                                   offsets_h.size());
  raft::update_device(offsets_d.data(), offsets_h.data(), offsets_h.size(),
                      cutils.stream);

  std::vector<int> parts_to_ranks_h;
  for (auto &rsp : utils.idxPartsToRanks) {
    int i = 0;
    for (int rank : utils.idxRanks) {
      if (rank == rsp->rank) {
        parts_to_ranks_h.push_back(i);
      }
      ++i;
    }
  }
  device_buffer<int> parts_to_ranks_d(cutils.alloc, cutils.stream,
                                      parts_to_ranks_h.size());
  raft::update_device(parts_to_ranks_d.data(), parts_to_ranks_h.data(),
                      parts_to_ranks_h.size(), cutils.stream);

  merge_labels_kernel<TPB_X><<<grid, blk, 0, cutils.stream>>>(
    output, knn_indices, unmerged_outputs, unmerged_knn_indices,
    offsets_d.data(), parts_to_ranks_d.data(), params.k, params.n_outputs,
    n_labels, utils.idxPartsToRanks.size(), utils.idxRanks.size());
}

void perform_local_operation(opg_knn_param &params, opg_knn_utils &utils,
                             cuda_utils &cutils, char32_t *outputs,
                             std::vector<float *> &probas_with_offsets,
                             char32_t *labels, size_t batch_size) {
  size_t n_labels = batch_size * params.k;

  if (params.knn_op == knn_operation::regression) {  // Regression
    std::vector<float *> y(params.n_outputs);
    for (int o = 0; o < params.n_outputs; o++) {
      y[o] = reinterpret_cast<float *>(labels) + (o * n_labels);
    }
    MLCommon::Selection::knn_regress<float, 32, true>(
      reinterpret_cast<float *>(outputs), nullptr, y, n_labels, batch_size,
      params.k, cutils.stream, cutils.internal_streams.data(),
      cutils.internal_streams.size());
  } else {
    std::vector<int *> y(params.n_outputs);
    for (int o = 0; o < params.n_outputs; o++) {
      y[o] = reinterpret_cast<int *>(labels) + (o * n_labels);
    }
    if (params.knn_op ==
        knn_operation::class_proba) {  // Class-probas operation
      MLCommon::Selection::class_probs<32, true>(
        probas_with_offsets, nullptr, y, n_labels, batch_size, params.k,
        *(params.uniq_labels), *(params.n_unique), cutils.alloc, cutils.stream,
        cutils.internal_streams.data(), cutils.internal_streams.size());
    } else if (params.knn_op ==
               knn_operation::classification) {  // Classification
      MLCommon::Selection::knn_classify<32, true>(
        reinterpret_cast<int *>(outputs), nullptr, y, n_labels, batch_size,
        params.k, *(params.uniq_labels), *(params.n_unique), cutils.alloc,
        cutils.stream, cutils.internal_streams.data(),
        cutils.internal_streams.size());
    }
  }
}

};  // namespace knn_common
};  // namespace opg
};  // namespace KNN
};  // namespace ML
