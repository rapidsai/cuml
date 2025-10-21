/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include <cuml/common/distance_type.hpp>
#include <cuml/common/logger.hpp>
#include <cuml/neighbors/knn_mg.hpp>

#include <cumlprims/opg/matrix/data.hpp>
#include <cumlprims/opg/matrix/part_descriptor.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/spatial/knn/knn.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cuvs/neighbors/knn_merge_parts.hpp>
#include <selection/knn.cuh>

#include <cstddef>
#include <memory>
#include <set>

namespace ML {
namespace KNN {
namespace opg {

namespace knn_common {

/**
 * The enumeration of KNN distributed operations
 */
enum knn_operation {
  knn,            /**< Simple KNN */
  classification, /**< KNN classification */
  class_proba,    /**< KNN classification probabilities */
  regression      /**< KNN regression */
};

/**
 * A structure to store parameters for distributed KNN
 */
template <typename in_t, typename ind_t, typename dist_t, typename out_t>
struct opg_knn_param {
  opg_knn_param(knn_operation knn_op,
                std::vector<Matrix::Data<in_t>*>* idx_data,
                Matrix::PartDescriptor* idx_desc,
                std::vector<Matrix::Data<in_t>*>* query_data,
                Matrix::PartDescriptor* query_desc,
                bool rowMajorIndex,
                bool rowMajorQuery,
                size_t k,
                size_t batch_size,
                bool verbose)
  {
    this->knn_op        = knn_op;
    this->idx_data      = idx_data;
    this->idx_desc      = idx_desc;
    this->query_data    = query_data;
    this->query_desc    = query_desc;
    this->rowMajorIndex = rowMajorIndex;
    this->rowMajorQuery = rowMajorQuery;
    this->k             = k;
    this->batch_size    = batch_size;
    this->verbose       = verbose;
  }

  knn_operation knn_op;                                   /**< Type of KNN distributed operation */
  std::vector<Matrix::Data<dist_t>*>* out_D    = nullptr; /**< KNN distances output array */
  std::vector<Matrix::Data<ind_t>*>* out_I     = nullptr; /**< KNN indices output array */
  std::vector<Matrix::Data<in_t>*>* idx_data   = nullptr; /**< Index input array */
  Matrix::PartDescriptor* idx_desc             = nullptr; /**< Descriptor for index input array */
  std::vector<Matrix::Data<in_t>*>* query_data = nullptr; /**< Query input array */
  Matrix::PartDescriptor* query_desc           = nullptr; /**< Descriptor for query input array */
  bool rowMajorIndex;                                     /**< Is index row major? */
  bool rowMajorQuery;                                     /**< Is query row major? */
  size_t k          = 0;                                  /**< Number of nearest neighbors */
  size_t batch_size = 0;                                  /**< Batch size */
  bool verbose;                                           /**< verbose */

  std::size_t n_outputs = 0;              /**< Number of outputs per query (cl&re) */
  std::vector<std::vector<out_t*>>* y;    /**< Labels input array (cl&re) */
  std::vector<Matrix::Data<out_t>*>* out; /**< KNN outputs output array (cl&re) */

  std::vector<int>* n_unique       = nullptr; /**< Number of unique labels (classification) */
  std::vector<out_t*>* uniq_labels = nullptr; /**< Unique labels (classification) */
  std::vector<std::vector<float*>>* probas =
    nullptr; /**< KNN classification probabilities output array (class-probas) */
};

template <typename in_t, typename ind_t, typename dist_t, typename out_t>
struct KNN_params : public opg_knn_param<in_t, ind_t, dist_t, out_t> {
  KNN_params(knn_operation knn_op,
             std::vector<Matrix::Data<in_t>*>* idx_data,
             Matrix::PartDescriptor* idx_desc,
             std::vector<Matrix::Data<in_t>*>* query_data,
             Matrix::PartDescriptor* query_desc,
             bool rowMajorIndex,
             bool rowMajorQuery,
             size_t k,
             size_t batch_size,
             bool verbose,
             std::vector<Matrix::Data<dist_t>*>* out_D,
             std::vector<Matrix::Data<ind_t>*>* out_I)
    : opg_knn_param<in_t, ind_t, dist_t, out_t>(knn_op,
                                                idx_data,
                                                idx_desc,
                                                query_data,
                                                query_desc,
                                                rowMajorIndex,
                                                rowMajorQuery,
                                                k,
                                                batch_size,
                                                verbose)
  {
    this->out_D = out_D;
    this->out_I = out_I;
  }
};

template <typename in_t, typename ind_t, typename dist_t, typename out_t>
struct KNN_RE_params : public opg_knn_param<in_t, ind_t, dist_t, out_t> {
  KNN_RE_params(knn_operation knn_op,
                std::vector<Matrix::Data<in_t>*>* idx_data,
                Matrix::PartDescriptor* idx_desc,
                std::vector<Matrix::Data<in_t>*>* query_data,
                Matrix::PartDescriptor* query_desc,
                bool rowMajorIndex,
                bool rowMajorQuery,
                size_t k,
                size_t batch_size,
                bool verbose,
                std::size_t n_outputs,
                std::vector<std::vector<out_t*>>* y,
                std::vector<Matrix::Data<out_t>*>* out)
    : opg_knn_param<in_t, ind_t, dist_t, out_t>(knn_op,
                                                idx_data,
                                                idx_desc,
                                                query_data,
                                                query_desc,
                                                rowMajorIndex,
                                                rowMajorQuery,
                                                k,
                                                batch_size,
                                                verbose)
  {
    this->n_outputs = n_outputs;
    this->y         = y;
    this->out       = out;
  }
};

template <typename in_t, typename ind_t, typename dist_t, typename out_t>
struct KNN_CL_params : public opg_knn_param<in_t, ind_t, dist_t, out_t> {
  KNN_CL_params(knn_operation knn_op,
                std::vector<Matrix::Data<in_t>*>* idx_data,
                Matrix::PartDescriptor* idx_desc,
                std::vector<Matrix::Data<in_t>*>* query_data,
                Matrix::PartDescriptor* query_desc,
                bool rowMajorIndex,
                bool rowMajorQuery,
                size_t k,
                size_t batch_size,
                bool verbose,
                std::size_t n_outputs,
                std::vector<std::vector<out_t*>>* y,
                std::vector<int>* n_unique,
                std::vector<out_t*>* uniq_labels,
                std::vector<Matrix::Data<out_t>*>* out,
                std::vector<std::vector<float*>>* probas)
    : opg_knn_param<in_t, ind_t, dist_t, out_t>(knn_op,
                                                idx_data,
                                                idx_desc,
                                                query_data,
                                                query_desc,
                                                rowMajorIndex,
                                                rowMajorQuery,
                                                k,
                                                batch_size,
                                                verbose)
  {
    this->n_outputs   = n_outputs;
    this->y           = y;
    this->n_unique    = n_unique;
    this->uniq_labels = uniq_labels;
    this->out         = out;
    this->probas      = probas;
  }
};

/**
 * A structure to store utilities for distributed KNN operations
 */
template <typename in_t, typename ind_t, typename dist_t, typename out_t>
struct opg_knn_work {
  opg_knn_work(opg_knn_param<in_t, ind_t, dist_t, out_t>& params, raft::handle_t& handle)
    : res_D(0, handle.get_stream()), res_I(0, handle.get_stream()), res(0, handle.get_stream())
  {
    this->my_rank           = handle.get_comms().get_rank();
    this->idxRanks          = params.idx_desc->uniqueRanks();
    this->idxPartsToRanks   = params.idx_desc->partsToRanks;
    this->local_idx_parts   = params.idx_desc->blocksOwnedBy(handle.get_comms().get_rank());
    this->queryPartsToRanks = params.query_desc->partsToRanks;
  }

  int my_rank;            /**< Rank of this worker */
  std::set<int> idxRanks; /**< Set of ranks having at least 1 index partition */
  std::vector<Matrix::RankSizePair*> idxPartsToRanks;   /**< Index parts to rank */
  std::vector<Matrix::RankSizePair*> local_idx_parts;   /**< List of index parts stored locally */
  std::vector<Matrix::RankSizePair*> queryPartsToRanks; /**< Query parts to rank */

  rmm::device_uvector<dist_t> res_D; /**< Temporary allocation to exchange distances */
  rmm::device_uvector<ind_t> res_I;  /**< Temporary allocation to exchange indices */
  rmm::device_uvector<out_t> res;    /**< Temporary allocation to exchange outputs (cl&re) */
};

/*!
 Main function, computes distributed KNN operation
 @param[in] params Parameters for distributed KNN operation
 @param[in] handle RAFT handle
 */
template <typename in_t, typename ind_t, typename dist_t, typename out_t>
void opg_knn(opg_knn_param<in_t, ind_t, dist_t, out_t>& params, raft::handle_t& handle)
{
  opg_knn_work<in_t, ind_t, dist_t, out_t> work(params, handle);

  ASSERT(params.k <= 1024, "k must be <= 1024");
  ASSERT(params.batch_size > 0, "max_batch_size must be > 0");
  ASSERT(params.k < params.idx_desc->M, "k must be less than the total number of query rows");
  for (Matrix::RankSizePair* rsp : work.idxPartsToRanks) {
    ASSERT(rsp->size >= params.k,
           "k must be <= the number of rows in the smallest index partition.");
  }

  int local_parts_completed = 0;
  // Loop through query parts for all ranks
  for (int i = 0; i < params.query_desc->totalBlocks(); i++) {  // For each query partitions
    Matrix::RankSizePair* partition = work.queryPartsToRanks[i];
    int part_rank                   = partition->rank;
    size_t part_n_rows              = partition->size;

    size_t total_batches     = raft::ceildiv(part_n_rows, params.batch_size);
    size_t total_n_processed = 0;

    // For each batch in a query partition
    for (std::size_t cur_batch = 0; cur_batch < total_batches; cur_batch++) {
      size_t cur_batch_size = params.batch_size;

      if (cur_batch == total_batches - 1)
        cur_batch_size = part_n_rows - (cur_batch * params.batch_size);

      if (work.my_rank == part_rank) CUML_LOG_DEBUG("Root Rank is %d", work.my_rank);

      /**
       * Root broadcasts batch to all other ranks
       */
      CUML_LOG_DEBUG("Rank %d: Performing Broadcast", work.my_rank);

      rmm::device_uvector<in_t> part_data(0, handle.get_stream());

      size_t batch_input_elms   = cur_batch_size * params.query_desc->N;
      size_t batch_input_offset = batch_input_elms * cur_batch;

      in_t* cur_query_ptr{nullptr};

      rmm::device_uvector<in_t> tmp_batch_buf(0, handle.get_stream());
      // current partition's owner rank broadcasts
      if (part_rank == work.my_rank) {
        Matrix::Data<in_t>* data = params.query_data->at(local_parts_completed);

        // If query is column major and total_batches > 0, create a
        // temporary buffer for the batch so that we can stack rows.
        if (!params.rowMajorQuery && total_batches > 1) {
          tmp_batch_buf.resize(batch_input_elms, handle.get_stream());
          for (std::size_t col_data = 0; col_data < params.query_desc->N; col_data++) {
            raft::copy(tmp_batch_buf.data() + (col_data * cur_batch_size),
                       data->ptr + ((col_data * part_n_rows) + total_n_processed),
                       cur_batch_size,
                       handle.get_stream());
          }
          cur_query_ptr = tmp_batch_buf.data();

        } else {
          cur_query_ptr = data->ptr + batch_input_offset;
        }

        // all other (index) ranks receive
      } else if (work.idxRanks.find(work.my_rank) != work.idxRanks.end()) {
        part_data.resize(batch_input_elms, handle.get_stream());
        cur_query_ptr = part_data.data();
      }

      bool my_rank_is_idx = work.idxRanks.find(work.my_rank) != work.idxRanks.end();

      /**
       * Send query to index partitions
       */
      if (work.my_rank == part_rank || my_rank_is_idx)
        broadcast_query(work, handle, part_rank, cur_query_ptr, batch_input_elms);

      if (my_rank_is_idx) {
        /**
         * All index ranks perform local KNN
         */
        CUML_LOG_DEBUG("Rank %d: Performing Local KNN", work.my_rank);

        size_t batch_knn_elms = params.k * cur_batch_size;

        if (params.knn_op != knn_operation::knn) {
          // No labels for KNN only operation
          work.res.resize(batch_knn_elms * params.n_outputs, handle.get_stream());
        }
        work.res_I.resize(batch_knn_elms, handle.get_stream());
        work.res_D.resize(batch_knn_elms, handle.get_stream());

        // Perform a local KNN search
        perform_local_knn(params, work, handle, cur_query_ptr, cur_batch_size);

        if (params.knn_op != knn_operation::knn) {
          // Get the right labels for indices obtained after a KNN merge
          copy_label_outputs_from_index_parts(params, work, handle, cur_batch_size);
        }
      }

      if (part_rank == work.my_rank || my_rank_is_idx) {
        /**
         * Ranks exchange results.
         * Each rank having index partition(s) sends
         * its local results (my_rank_is_idx)
         * Additionally the owner of currently processed query partition
         * receives and performs a reduce even if it has
         * no index partition (part_rank == my_rank)
         */
        CUML_LOG_DEBUG("Rank %d: Exchanging results", work.my_rank);
        exchange_results(params, work, handle, part_rank, cur_batch_size);
      }

      /**
       * Root rank performs local reduce
       */
      if (part_rank == work.my_rank) {
        CUML_LOG_DEBUG("Rank %d: Performing Reduce", work.my_rank);

        // Reduce all local results to a global result for a given query batch
        reduce(params, work, handle, local_parts_completed, total_n_processed, cur_batch_size);

        CUML_LOG_DEBUG("Rank %d: Finished Reduce", work.my_rank);
      }

      total_n_processed += cur_batch_size;
    }

    if (work.my_rank == part_rank) local_parts_completed++;
  }
};

/*!
 Broadcast query batch across all the workers
 @param[in] params Parameters for distributed KNN operation
 @param[in] handle RAFT handle
 @param[in] part_rank Rank of currently processed query batch
 @param[in] broadcast Pointer to broadcast
 @param[in] broadcast_size Size of broadcast
 */
template <typename in_t, typename ind_t, typename dist_t, typename out_t>
void broadcast_query(opg_knn_work<in_t, ind_t, dist_t, out_t>& work,
                     raft::handle_t& handle,
                     int part_rank,
                     in_t* broadcast,
                     size_t broadcast_size)
{
  int request_idx = 0;
  std::vector<raft::comms::request_t> requests;
  if (part_rank == work.my_rank) {  // Either broadcast to other workers
    int idx_rank_size = work.idxRanks.size();
    if (work.idxRanks.find(work.my_rank) != work.idxRanks.end()) { --idx_rank_size; }

    requests.resize(idx_rank_size);

    for (int rank : work.idxRanks) {
      if (rank != work.my_rank) {
        handle.get_comms().isend(broadcast, broadcast_size, rank, 0, requests.data() + request_idx);
        ++request_idx;
      }
    }

  } else {  // Or receive from broadcaster
    requests.resize(1);
    handle.get_comms().irecv(
      broadcast, broadcast_size, part_rank, 0, requests.data() + request_idx);
    ++request_idx;
  }

  handle.get_comms().waitall(requests.size(), requests.data());
}

/*!
 Perform a local KNN search for a given query batch
 @param[in] params Parameters for distributed KNN operation
 @param[in] work Current work for distributed KNN
 @param[in] handle RAFT handle
 @param[in] query Pointer to query
 @param[in] query_size Size of query
 */
template <typename in_t, typename ind_t, typename dist_t, typename out_t>
void perform_local_knn(opg_knn_param<in_t, ind_t, dist_t, out_t>& params,
                       opg_knn_work<in_t, ind_t, dist_t, out_t>& work,
                       raft::handle_t& handle,
                       in_t* query,
                       size_t query_size)
{
  std::vector<in_t*> ptrs(params.idx_data->size());
  std::vector<int> sizes(params.idx_data->size());

  for (std::size_t cur_idx = 0; cur_idx < params.idx_data->size(); cur_idx++) {
    ptrs[cur_idx]  = params.idx_data->at(cur_idx)->ptr;
    sizes[cur_idx] = work.local_idx_parts[cur_idx]->size;
  }

  // Offset nearest neighbor index matrix by partition indices
  std::vector<size_t> start_indices = params.idx_desc->startIndices(work.my_rank);
  // PartDescriptor uses size_t while FAISS uses int64_t
  // so we need to do a quick conversion.
  std::vector<int64_t> start_indices_long;
  for (size_t start_index : start_indices)
    start_indices_long.push_back((int64_t)start_index);

  // ID ranges need to be offset by each local partition's
  // starting indices.
  brute_force_knn(handle,
                  ptrs,
                  sizes,
                  params.idx_desc->N,
                  query,
                  query_size,
                  work.res_I.data(),
                  work.res_D.data(),
                  params.k,
                  params.rowMajorIndex,
                  params.rowMajorQuery,
                  distance::DistanceType::L2SqrtExpanded,
                  2.0f,
                  &start_indices_long);
  handle.sync_stream(handle.get_stream());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
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
template <int TPB_X, typename ind_t, typename out_t>
CUML_KERNEL void copy_label_outputs_from_index_parts_kernel(out_t* out,
                                                            ind_t* knn_indices,
                                                            out_t** parts,
                                                            uint64_t* offsets,
                                                            size_t cur_batch_size,
                                                            int n_parts,
                                                            int n_labels)
{
  uint64_t i = (blockIdx.x * TPB_X) + threadIdx.x;
  if (i >= n_labels) return;
  uint64_t nn_idx = knn_indices[i];
  int part_idx    = 0;
  for (; part_idx < n_parts && nn_idx >= offsets[part_idx]; part_idx++) {}
  part_idx        = std::min(std::max(0, part_idx - 1), n_parts - 1);
  uint64_t offset = nn_idx - offsets[part_idx];
  out[i]          = parts[part_idx][offset];
}

/*!
 Get the right labels for indices obtained after a KNN merge
 @param[in] params Parameters for distributed KNN operation
 @param[in] work Current work for distributed KNN
 @param[in] handle RAFT handle
 @param[in] batch_size Batch size
 */
template <typename in_t, typename ind_t, typename dist_t, typename out_t>
void copy_label_outputs_from_index_parts(opg_knn_param<in_t, ind_t, dist_t, out_t>& params,
                                         opg_knn_work<in_t, ind_t, dist_t, out_t>& work,
                                         raft::handle_t& handle,
                                         size_t batch_size)
{
  const int TPB_X = 256;
  int n_labels    = batch_size * params.k;
  dim3 grid(raft::ceildiv(n_labels, TPB_X));
  dim3 blk(TPB_X);

  uint64_t offset = 0;
  std::vector<uint64_t> offsets_h;
  for (auto& rsp : work.idxPartsToRanks) {
    if (rsp->rank == work.my_rank) { offsets_h.push_back(offset); }
    offset += rsp->size;
  }
  std::size_t n_parts = offsets_h.size();
  rmm::device_uvector<uint64_t> offsets_d(n_parts, handle.get_stream());
  raft::update_device(offsets_d.data(), offsets_h.data(), n_parts, handle.get_stream());

  std::vector<out_t*> parts_h(n_parts);
  rmm::device_uvector<out_t*> parts_d(n_parts, handle.get_stream());
  for (std::size_t o = 0; o < params.n_outputs; o++) {
    for (std::size_t p = 0; p < n_parts; p++) {
      parts_h[p] = params.y->at(p)[o];
    }
    raft::update_device(parts_d.data(), parts_h.data(), n_parts, handle.get_stream());

    copy_label_outputs_from_index_parts_kernel<TPB_X, ind_t, out_t>
      <<<grid, blk, 0, handle.get_stream()>>>(work.res.data() + (o * n_labels),
                                              work.res_I.data(),
                                              parts_d.data(),
                                              offsets_d.data(),
                                              batch_size,
                                              n_parts,
                                              n_labels);
  }
  handle.sync_stream(handle.get_stream());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/*!
 Exchange results of local KNN search and operation for a given query batch
 All non-root index ranks send the results for the current
 query batch to the root rank for the batch.
 @param[in] params Parameters for distributed KNN operation
 @param[in] work Current work for distributed KNN
 @param[in] handle RAFT handle
 @param[in] part_rank Rank of currently processed query batch
 @param[in] batch_size Batch size
 */
template <typename in_t, typename ind_t, typename dist_t, typename out_t>
void exchange_results(opg_knn_param<in_t, ind_t, dist_t, out_t>& params,
                      opg_knn_work<in_t, ind_t, dist_t, out_t>& work,
                      raft::handle_t& handle,
                      int part_rank,
                      size_t batch_size)
{
  size_t batch_elms = batch_size * params.k;

  int request_idx = 0;
  std::vector<raft::comms::request_t> requests;
  if (part_rank != work.my_rank) {  // Either send local KNN results
    requests.resize(2);
    handle.get_comms().isend(
      work.res_I.data(), batch_elms, part_rank, 0, requests.data() + request_idx);
    ++request_idx;

    handle.get_comms().isend(
      work.res_D.data(), batch_elms, part_rank, 0, requests.data() + request_idx);
    ++request_idx;

    if (params.knn_op != knn_operation::knn) {
      requests.resize(2 + params.n_outputs);
      for (std::size_t o = 0; o < params.n_outputs; o++) {
        handle.get_comms().isend(work.res.data() + (o * batch_elms),
                                 batch_elms,
                                 part_rank,
                                 0,
                                 requests.data() + request_idx);
        ++request_idx;
      }
    }
  } else {  // Or, as the owner of currently processed query batch,
    // receive results from other workers for reduce
    bool part_rank_is_idx = work.idxRanks.find(part_rank) != work.idxRanks.end();
    size_t idx_rank_size  = work.idxRanks.size();

    // if root rank is an index, it will already have
    // query data, so no need to receive from it.
    work.res_I.resize(batch_elms * idx_rank_size, handle.get_stream());
    work.res_D.resize(batch_elms * idx_rank_size, handle.get_stream());

    if (params.knn_op != knn_operation::knn) {
      work.res.resize(batch_elms * params.n_outputs * idx_rank_size, handle.get_stream());
    }

    if (part_rank_is_idx) {
      /**
       * If this worker (in charge of reduce),
       * has some local results as well,
       * copy them at right location
       */
      --idx_rank_size;
      int i = 0;
      for (int rank : work.idxRanks) {
        if (rank == work.my_rank) {
          size_t batch_offset = batch_elms * i;

          // Indices and distances are stored in rank order
          raft::copy_async(
            work.res_I.data() + batch_offset, work.res_I.data(), batch_elms, handle.get_stream());
          raft::copy_async(
            work.res_D.data() + batch_offset, work.res_D.data(), batch_elms, handle.get_stream());

          if (params.knn_op != knn_operation::knn) {
            rmm::device_uvector<out_t> tmp_res(params.n_outputs * batch_elms, handle.get_stream());
            raft::copy_async(tmp_res.data(), work.res.data(), tmp_res.size(), handle.get_stream());

            for (std::size_t o = 0; o < params.n_outputs; ++o) {
              // Outputs are stored in target order and then in rank order
              raft::copy_async(
                work.res.data() + (o * work.idxRanks.size() * batch_elms) + batch_offset,
                tmp_res.data() + (o * batch_elms),
                batch_elms,
                handle.get_stream());
            }
          }
          handle.sync_stream(handle.get_stream());
          break;
        }
        i++;
      }
    }

    size_t request_size = 2 * idx_rank_size;
    if (params.knn_op != knn_operation::knn) request_size = (2 + params.n_outputs) * idx_rank_size;
    requests.resize(request_size);

    int num_received = 0;
    for (int rank : work.idxRanks) {
      if (rank != work.my_rank) {
        size_t batch_offset = batch_elms * num_received;

        // Indices and distances are stored in rank order
        handle.get_comms().irecv(
          work.res_I.data() + batch_offset, batch_elms, rank, 0, requests.data() + request_idx);
        ++request_idx;
        handle.get_comms().irecv(
          work.res_D.data() + batch_offset, batch_elms, rank, 0, requests.data() + request_idx);
        ++request_idx;

        if (params.knn_op != knn_operation::knn) {
          for (std::size_t o = 0; o < params.n_outputs; o++) {
            // Outputs are stored in target order and then in rank order
            out_t* r = work.res.data() + (o * work.idxRanks.size() * batch_elms) + batch_offset;
            handle.get_comms().irecv(r, batch_elms, rank, 0, requests.data() + request_idx);
            ++request_idx;
          }
        }
      }
      if (rank != work.my_rank || part_rank_is_idx) {
        /**
         * Increase index for each new reception
         * Also increase index when the worker doing a reduce operation
         * has some index data (previously copied at right location).
         */
        ++num_received;
      }
    }
  }

  handle.get_comms().waitall(requests.size(), requests.data());
}

/*!
 Reduce all local results to a global result for a given query batch
 @param[in] params Parameters for distributed KNN operation
 @param[in] work Current work for distributed KNN
 @param[in] handle RAFT handle
 @param[in] part_idx Partition index of query batch
 @param[in] processed_in_part Number of queries already processed in part (serves as offset)
 @param[in] batch_size Batch size
 */
template <typename in_t,
          typename ind_t,
          typename dist_t,
          typename out_t,
          typename trans_t = int64_t>
void reduce(opg_knn_param<in_t, ind_t, dist_t, out_t>& params,
            opg_knn_work<in_t, ind_t, dist_t, out_t>& work,
            raft::handle_t& handle,
            int part_idx,
            size_t processed_in_part,
            size_t batch_size)
{
  rmm::device_uvector<trans_t> trans(work.idxRanks.size(), handle.get_stream());
  RAFT_CUDA_TRY(
    cudaMemsetAsync(trans.data(), 0, work.idxRanks.size() * sizeof(trans_t), handle.get_stream()));

  size_t batch_offset = processed_in_part * params.k;

  ind_t* indices    = nullptr;
  dist_t* distances = nullptr;

  rmm::device_uvector<ind_t> indices_b(0, handle.get_stream());
  rmm::device_uvector<dist_t> distances_b(0, handle.get_stream());

  if (params.knn_op == knn_operation::knn) {
    indices   = params.out_I->at(part_idx)->ptr + batch_offset;
    distances = params.out_D->at(part_idx)->ptr + batch_offset;
  } else {
    indices_b.resize(batch_size * params.k, handle.get_stream());
    distances_b.resize(batch_size * params.k, handle.get_stream());
    indices   = indices_b.data();
    distances = distances_b.data();
  }

  // Merge all KNN local results
  cuvs::neighbors::knn_merge_parts(
    handle,
    raft::make_device_matrix_view<const dist_t, int64_t>(
      work.res_D.data(), batch_size * work.idxRanks.size(), params.k),
    raft::make_device_matrix_view<const ind_t, int64_t>(
      work.res_I.data(), batch_size * work.idxRanks.size(), params.k),
    raft::make_device_matrix_view<dist_t, int64_t>(distances, batch_size, params.k),
    raft::make_device_matrix_view<ind_t, int64_t>(indices, batch_size, params.k),
    raft::make_device_vector_view<trans_t>(trans.data(), trans.size()));
  handle.sync_stream(handle.get_stream());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  if (params.knn_op != knn_operation::knn) {
    rmm::device_uvector<out_t> merged_outputs_b(params.n_outputs * batch_size * params.k,
                                                handle.get_stream());
    // Get the right labels for indices obtained after local KNN searches
    merge_labels(params,
                 work,
                 handle,
                 merged_outputs_b.data(),
                 indices,
                 work.res.data(),
                 work.res_I.data(),
                 batch_size);

    out_t* outputs = nullptr;
    std::vector<float*> probas_with_offsets;

    if (params.knn_op != knn_operation::class_proba) {
      outputs = params.out->at(part_idx)->ptr + (processed_in_part * params.n_outputs);
    } else {
      std::vector<float*>& probas_part = params.probas->at(part_idx);
      for (std::size_t i = 0; i < params.n_outputs; i++) {
        float* ptr           = probas_part[i];
        int n_unique_classes = params.n_unique->at(i);
        probas_with_offsets.push_back(ptr + (processed_in_part * n_unique_classes));
      }
    }

    // Perform final classification, regression or class-proba operation
    perform_local_operation(
      params, work, handle, outputs, probas_with_offsets, merged_outputs_b.data(), batch_size);

    handle.sync_stream(handle.get_stream());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
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
template <int TPB_X, typename dist_t, typename out_t>
CUML_KERNEL void merge_labels_kernel(out_t* outputs,
                                     dist_t* knn_indices,
                                     out_t* unmerged_outputs,
                                     dist_t* unmerged_knn_indices,
                                     size_t* offsets,
                                     int* parts_to_ranks,
                                     int nearest_neighbors,
                                     int n_outputs,
                                     int n_labels,
                                     int n_parts,
                                     int n_ranks)
{
  uint64_t i = (blockIdx.x * TPB_X) + threadIdx.x;
  if (i >= n_labels) return;
  uint64_t nn_idx = knn_indices[i];
  int part_idx    = 0;
  for (; part_idx < n_parts && nn_idx >= offsets[part_idx]; part_idx++) {}
  part_idx         = std::min(std::max(0, part_idx - 1), n_parts - 1);
  int rank_idx     = parts_to_ranks[part_idx];
  int inbatch_idx  = i / nearest_neighbors;
  uint64_t elm_idx = (rank_idx * n_labels) + inbatch_idx * nearest_neighbors;
  for (int k = 0; k < nearest_neighbors; k++) {
    if (nn_idx == unmerged_knn_indices[elm_idx + k]) {
      for (int o = 0; o < n_outputs; o++) {
        outputs[(o * n_labels) + i] = unmerged_outputs[(o * n_ranks * n_labels) + elm_idx + k];
      }
      return;
    }
  }
}

/*!
 Get the right labels for indices obtained after local KNN searches
 @param[in] params Parameters for distributed KNN operation
 @param[in] work Current work for distributed KNN
 @param[in] handle RAFT handle
 @param[out] output KNN outputs output array
 @param[out] knn_indices KNN class-probas output array (class-proba only)
 @param[in] unmerged_outputs KNN labels input array
 @param[in] unmerged_knn_indices Batch size
 @param[in] batch_size Batch size
 */
template <typename opg_knn_param_t, typename opg_knn_work_t, typename ind_t, typename out_t>
void merge_labels(opg_knn_param_t& params,
                  opg_knn_work_t& work,
                  raft::handle_t& handle,
                  out_t* output,
                  ind_t* knn_indices,
                  out_t* unmerged_outputs,
                  ind_t* unmerged_knn_indices,
                  int batch_size)
{
  const int TPB_X = 256;
  int n_labels    = batch_size * params.k;
  dim3 grid(raft::ceildiv(n_labels, TPB_X));
  dim3 blk(TPB_X);

  int offset = 0;
  std::vector<uint64_t> offsets_h;
  for (auto& rsp : work.idxPartsToRanks) {
    offsets_h.push_back(offset);
    offset += rsp->size;
  }
  rmm::device_uvector<uint64_t> offsets_d(offsets_h.size(), handle.get_stream());
  raft::update_device(offsets_d.data(), offsets_h.data(), offsets_h.size(), handle.get_stream());

  std::vector<int> parts_to_ranks_h;
  for (auto& rsp : work.idxPartsToRanks) {
    int i = 0;
    for (int rank : work.idxRanks) {
      if (rank == rsp->rank) { parts_to_ranks_h.push_back(i); }
      ++i;
    }
  }
  rmm::device_uvector<int> parts_to_ranks_d(parts_to_ranks_h.size(), handle.get_stream());
  raft::update_device(
    parts_to_ranks_d.data(), parts_to_ranks_h.data(), parts_to_ranks_h.size(), handle.get_stream());

  merge_labels_kernel<TPB_X><<<grid, blk, 0, handle.get_stream()>>>(output,
                                                                    knn_indices,
                                                                    unmerged_outputs,
                                                                    unmerged_knn_indices,
                                                                    offsets_d.data(),
                                                                    parts_to_ranks_d.data(),
                                                                    params.k,
                                                                    params.n_outputs,
                                                                    n_labels,
                                                                    work.idxPartsToRanks.size(),
                                                                    work.idxRanks.size());
}

/*!
 Perform final classification, regression or class-proba operation for a given query batch
 @param[in] params Parameters for distributed KNN operation
 @param[in] work Current work for distributed KNN
 @param[in] handle RAFT handle
 @param[out] outputs KNN outputs output array
 @param[out] probas_with_offsets KNN class-probas output array (class-proba only)
 @param[in] labels KNN labels input array
 @param[in] batch_size Batch size
 */
template <typename in_t,
          typename ind_t,
          typename dist_t,
          typename out_t,
          typename std::enable_if<std::is_floating_point<out_t>::value>::type* = nullptr>
void perform_local_operation(opg_knn_param<in_t, ind_t, dist_t, out_t>& params,
                             opg_knn_work<in_t, ind_t, dist_t, out_t>& work,
                             raft::handle_t& handle,
                             out_t* outputs,
                             std::vector<float*>& probas_with_offsets,
                             out_t* labels,
                             size_t batch_size)
{
  size_t n_labels = batch_size * params.k;
  std::vector<out_t*> y(params.n_outputs);
  for (std::size_t o = 0; o < params.n_outputs; o++) {
    y[o] = reinterpret_cast<out_t*>(labels) + (o * n_labels);
  }

  MLCommon::Selection::knn_regress<float, 32, true>(
    handle, outputs, nullptr, y, n_labels, batch_size, params.k);
}

/*!
 Perform final classification, regression or class-proba operation for a given query batch
 @param[in] params Parameters for distributed KNN operation
 @param[in] work Current work for distributed KNN
 @param[in] handle RAFT handle
 @param[out] outputs KNN outputs output array
 @param[out] probas_with_offsets KNN class-probas output array (class-proba only)
 @param[in] labels KNN labels input array
 @param[in] batch_size Batch size
 */
template <typename in_t,
          typename ind_t,
          typename dist_t,
          typename out_t,
          typename std::enable_if<std::is_integral<out_t>::value>::type* = nullptr>
void perform_local_operation(opg_knn_param<in_t, ind_t, dist_t, out_t>& params,
                             opg_knn_work<in_t, ind_t, dist_t, out_t>& work,
                             raft::handle_t& handle,
                             out_t* outputs,
                             std::vector<float*>& probas_with_offsets,
                             out_t* labels,
                             size_t batch_size)
{
  size_t n_labels = batch_size * params.k;
  std::vector<out_t*> y(params.n_outputs);
  for (std::size_t o = 0; o < params.n_outputs; o++) {
    y[o] = reinterpret_cast<out_t*>(labels) + (o * n_labels);
  }

  switch (params.knn_op) {
    case knn_operation::classification:
      MLCommon::Selection::knn_classify<32, true>(handle,
                                                  outputs,
                                                  nullptr,
                                                  y,
                                                  n_labels,
                                                  batch_size,
                                                  params.k,
                                                  *(params.uniq_labels),
                                                  *(params.n_unique));
      break;
    case knn_operation::class_proba:
      MLCommon::Selection::class_probs<32, true>(handle,
                                                 probas_with_offsets,
                                                 nullptr,
                                                 y,
                                                 n_labels,
                                                 batch_size,
                                                 params.k,
                                                 *(params.uniq_labels),
                                                 *(params.n_unique));
      break;
    default: CUML_LOG_DEBUG("Not a valid operation.");
  }
}

};  // namespace knn_common
};  // namespace opg
};  // namespace KNN
};  // namespace ML
