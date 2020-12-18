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

enum knn_operation { knn, classification, class_proba, regression };

struct opg_knn_param {
  knn_operation knn_op;
  std::vector<Matrix::Data<int64_t> *> *out_I = nullptr;
  std::vector<Matrix::floatData_t *> *out_D = nullptr;
  std::vector<Matrix::floatData_t *> *idx_data = nullptr;
  Matrix::PartDescriptor *idx_desc = nullptr;
  std::vector<Matrix::floatData_t *> *query_data = nullptr;
  Matrix::PartDescriptor *query_desc = nullptr;
  bool rowMajorIndex;
  bool rowMajorQuery;
  size_t k = 0;
  int n_outputs = 0;
  size_t batch_size = 0;
  bool verbose;
  std::vector<std::vector<float *>> *probas = nullptr;
  std::vector<int *> *uniq_labels = nullptr;
  std::vector<int> *n_unique = nullptr;

  union labels_data {
    std::vector<std::vector<int *>> *i;
    std::vector<std::vector<float *>> *f;
  };

  union outputs_data {
    std::vector<Matrix::Data<int> *> *i;
    std::vector<Matrix::Data<float> *> *f;
  };

  outputs_data out;
  labels_data y;
};

struct cuda_utils {
  cuda_utils(raft::handle_t &handle) {
    this->alloc = handle.get_device_allocator();
    this->stream = handle.get_stream();
    this->comm = &(handle.get_comms());  //communicator_ is a private attribute
    this->n_internal_streams = handle.get_num_internal_streams();
    this->internal_streams_.resize(this->n_internal_streams);
    for (int i = 0; i < this->n_internal_streams; i++) {
      internal_streams_[i] = handle.get_internal_stream(i);
    }
    this->internal_streams = internal_streams_.data();
  };
  std::shared_ptr<deviceAllocator> alloc;
  cudaStream_t stream;
  const raft::comms::comms_t *comm;
  cudaStream_t *internal_streams;
  std::vector<cudaStream_t> internal_streams_;
  int n_internal_streams;
};

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

  int my_rank;
  std::set<int> idxRanks;
  std::vector<Matrix::RankSizePair *> idxPartsToRanks;
  std::vector<Matrix::RankSizePair *> local_idx_parts;
  std::vector<Matrix::RankSizePair *> queryPartsToRanks;

  device_buffer<int64_t> *res_I;
  device_buffer<float> *res_D;
  device_buffer<char32_t> *res;
};

void opg_knn(opg_knn_param &params, cuda_utils &cutils);

void broadcast_query(opg_knn_utils &utils, cuda_utils &cutils, float *query,
                     size_t batch_input_elms, int part_rank);

void perform_local_knn(opg_knn_param &params, opg_knn_utils &utils,
                       cuda_utils &cutils, size_t cur_batch_size,
                       float *cur_query_ptr);

void perform_local_operation(opg_knn_param &params, opg_knn_utils &utils,
                             cuda_utils &cutils, char32_t *out,
                             int64_t *knn_indices, char32_t *labels,
                             size_t cur_batch_size,
                             std::vector<float *> &probas_with_offsets);

void exchange_results(opg_knn_param &params, opg_knn_utils &utils,
                      cuda_utils &cutils, int part_rank, size_t cur_batch_size,
                      int local_parts_completed);

void reduce(opg_knn_param &params, opg_knn_utils &utils, cuda_utils &cutils,
            size_t cur_batch_size, int local_parts_completed, int cur_batch,
            size_t total_n_processed);

};  // namespace knn_common
};  // namespace opg
};  // namespace KNN
};  // namespace ML