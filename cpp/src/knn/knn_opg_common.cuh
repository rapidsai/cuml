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

#pragma once

#include <cuml/neighbors/knn_mg.hpp>
#include <selection/knn.cuh>

#include <common/cumlHandle.hpp>

#include <common/device_buffer.hpp>
#include <cuml/common/cuml_allocator.hpp>
#include <raft/comms/comms.hpp>

#include <set>

#include <raft/cuda_utils.cuh>

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
struct opg_knn_param {
  union labels_data {
    std::vector<std::vector<int *>> *i;
    std::vector<std::vector<float *>> *f;
  };

  union outputs_data {
    std::vector<Matrix::Data<int> *> *i;
    std::vector<Matrix::Data<float> *> *f;
  };

  knn_operation knn_op; /**< Type of KNN distributed operation */
  std::vector<Matrix::Data<int64_t> *> *out_I =
    nullptr; /**< KNN indices output array */
  std::vector<Matrix::floatData_t *> *out_D =
    nullptr; /**< KNN distances output array */
  std::vector<Matrix::floatData_t *> *idx_data =
    nullptr; /**< Index input array */
  Matrix::PartDescriptor *idx_desc =
    nullptr; /**< Descriptor for index input array */
  std::vector<Matrix::floatData_t *> *query_data =
    nullptr; /**< Query input array */
  Matrix::PartDescriptor *query_desc =
    nullptr;             /**< Descriptor for query input array */
  bool rowMajorIndex;    /**< Is index row major? */
  bool rowMajorQuery;    /**< Is query row major? */
  size_t k = 0;          /**< Number of nearest neighbors */
  size_t batch_size = 0; /**< Batch size */
  bool verbose;          /**< verbose */

  labels_data y;     /**< Labels input array (cl&re) */
  int n_outputs = 0; /**< Number of outputs per query (cl&re) */
  outputs_data out;  /**< KNN outputs output array (cl&re) */

  std::vector<int *> *uniq_labels =
    nullptr; /**< Unique labels (classification) */
  std::vector<int> *n_unique =
    nullptr; /**< Number of unique labels (classification) */

  std::vector<std::vector<float *>> *probas =
    nullptr; /**< KNN classification probabilities output array (class-probas) */
};

/**
 * A structure to store utilities for CUDA and RAFT comms
 */
struct cuda_utils {
  cuda_utils(raft::handle_t &handle) {
    this->alloc = handle.get_device_allocator();
    this->stream = handle.get_stream();
    this->comm = &(handle.get_comms());  //communicator_ is a private attribute
    size_t n_internal_streams = handle.get_num_internal_streams();
    this->internal_streams.resize(n_internal_streams);
    for (int i = 0; i < n_internal_streams; i++) {
      internal_streams[i] = handle.get_internal_stream(i);
    }
  };
  std::shared_ptr<deviceAllocator> alloc; /**< RMM alloc */
  cudaStream_t stream;                    /**< CUDA user stream */
  const raft::comms::comms_t *comm;       /**< RAFT comms handle */
  std::vector<cudaStream_t>
    internal_streams; /**< Vector of CUDA internal streams */
};

/**
 * A structure to store utilities for distributed KNN operations
 */
struct opg_knn_utils {
  opg_knn_utils(opg_knn_param &params, cuda_utils &cutils) {
    this->my_rank = cutils.comm->get_rank();
    this->idxRanks = params.idx_desc->uniqueRanks();
    this->idxPartsToRanks = params.idx_desc->partsToRanks;
    this->local_idx_parts =
      params.idx_desc->blocksOwnedBy(cutils.comm->get_rank());
    this->queryPartsToRanks = params.query_desc->partsToRanks;

    this->res_I = new device_buffer<int64_t>(cutils.alloc, cutils.stream);
    this->res_D = new device_buffer<float>(cutils.alloc, cutils.stream);
    this->res = new device_buffer<char32_t>(cutils.alloc, cutils.stream);
  };

  ~opg_knn_utils() {
    delete res_I;
    delete res_D;
    delete res;
  };

  int my_rank;            /**< Rank of this worker */
  std::set<int> idxRanks; /**< Set of ranks having at least 1 index partition */
  std::vector<Matrix::RankSizePair *>
    idxPartsToRanks; /**< Index parts to rank */
  std::vector<Matrix::RankSizePair *>
    local_idx_parts; /**< List of index parts stored locally */
  std::vector<Matrix::RankSizePair *>
    queryPartsToRanks; /**< Query parts to rank */

  device_buffer<int64_t>
    *res_I; /**< Temporary allocation to exchange indices */
  device_buffer<float>
    *res_D; /**< Temporary allocation to exchange distances */
  device_buffer<char32_t>
    *res; /**< Temporary allocation to exchange outputs (cl&re) */
};

/*!
 Main function, computes distributed KNN operation
 @param[in] params Parameters for distrbuted KNN operation
 @param[in] cutils Utilities for CUDA and RAFT comms
 */
void opg_knn(opg_knn_param &params, cuda_utils &cutils);

/*!
 Broadcast query batch accross all the workers
 @param[in] params Parameters for distrbuted KNN operation
 @param[in] cutils Utilities for CUDA and RAFT comms
 @param[in] part_rank Rank of currently processed query batch
 @param[in] broadcast Pointer to broadcast
 @param[in] broadcast_size Size of broadcast
 */
void broadcast_query(opg_knn_utils &utils, cuda_utils &cutils, int part_rank,
                     float *broadcast, size_t broadcast_size);

/*!
 Perform a local KNN search for a given query batch
 @param[in] params Parameters for distrbuted KNN operation
 @param[in] utils Utilities for distributed KNN
 @param[in] cutils Utilities for CUDA and RAFT comms
 @param[in] query Pointer to query
 @param[in] query_size Size of query
 */
void perform_local_knn(opg_knn_param &params, opg_knn_utils &utils,
                       cuda_utils &cutils, float *query, size_t query_size);

/*!
 Get the right labels for indices obtained after a KNN merge
 @param[in] params Parameters for distrbuted KNN operation
 @param[in] utils Utilities for distributed KNN
 @param[in] cutils Utilities for CUDA and RAFT comms
 @param[in] batch_size Batch size
 */
void copy_label_outputs_from_index_parts(opg_knn_param &params,
                                         opg_knn_utils &utils,
                                         cuda_utils &cutils, size_t batch_size);

/*!
 Exchange results of local KNN search and operation for a given query batch
 @param[in] params Parameters for distrbuted KNN operation
 @param[in] utils Utilities for distributed KNN
 @param[in] cutils Utilities for CUDA and RAFT comms
 @param[in] part_rank Rank of currently processed query batch
 @param[in] batch_size Batch size
 */
void exchange_results(opg_knn_param &params, opg_knn_utils &utils,
                      cuda_utils &cutils, int part_rank, size_t cur_batch_size);

/*!
 Reduce all local results to a global result for a given query batch
 @param[in] params Parameters for distrbuted KNN operation
 @param[in] utils Utilities for distributed KNN
 @param[in] cutils Utilities for CUDA and RAFT comms
 @param[in] part_idx Partition index of query batch
 @param[in] processed_in_part Number of queries already processed in part (serves as offset)
 @param[in] batch_size Batch size
 */
void reduce(opg_knn_param &params, opg_knn_utils &utils, cuda_utils &cutils,
            int part_idx, size_t processed_in_part, size_t batch_size);

/*!
 Get the right labels for indices obtained after local KNN searches
 @param[in] params Parameters for distrbuted KNN operation
 @param[in] utils Utilities for distributed KNN
 @param[in] cutils Utilities for CUDA and RAFT comms
 @param[out] output KNN outputs output array
 @param[out] knn_indices KNN class-probas output array (class-proba only)
 @param[in] unmerged_outputs KNN labels input array
 @param[in] unmerged_knn_indices Batch size
 @param[in] batch_size Batch size
 */
void merge_labels(opg_knn_param &params, opg_knn_utils &utils,
                  cuda_utils &cutils, char32_t *output, int64_t *knn_indices,
                  char32_t *unmerged_outputs, int64_t *unmerged_knn_indices,
                  int batch_size);

/*!
 Perform final classification, regression or class-proba operation for a given query batch
 @param[in] params Parameters for distrbuted KNN operation
 @param[in] utils Utilities for distributed KNN
 @param[in] cutils Utilities for CUDA and RAFT comms
 @param[out] outputs KNN outputs output array
 @param[out] probas_with_offsets KNN class-probas output array (class-proba only)
 @param[in] labels KNN labels input array
 @param[in] batch_size Batch size
 */
void perform_local_operation(opg_knn_param &params, opg_knn_utils &utils,
                             cuda_utils &cutils, char32_t *outputs,
                             std::vector<float *> &probas_with_offsets,
                             char32_t *labels, size_t batch_size);

};  // namespace knn_common
};  // namespace opg
};  // namespace KNN
};  // namespace ML