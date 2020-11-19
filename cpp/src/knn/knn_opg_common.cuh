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
             std::vector<int> *n_unique = nullptr, bool probas_only = false);

template <typename T>
void reduce(raft::handle_t &handle, std::vector<Matrix::Data<T> *> *out,
            std::vector<Matrix::Data<int64_t> *> *out_I,
            std::vector<Matrix::floatData_t *> *out_D, device_buffer<T> &res,
            device_buffer<int64_t> &res_I, device_buffer<float> &res_D,
            Matrix::PartDescriptor &index_desc, size_t cur_batch_size, int k,
            int n_outputs, int local_parts_completed, int cur_batch,
            size_t total_n_processed, std::set<int> idxRanks, int my_rank,
            bool probas_only = false,
            std::vector<std::vector<float *>> *probas = nullptr,
            std::vector<int *> *uniq_labels = nullptr,
            std::vector<int> *n_unique = nullptr);

void broadcast_query(float *query, size_t batch_input_elms, int part_rank,
                     std::set<int> idxRanks, const raft::comms::comms_t &comm,
                     cudaStream_t stream);

template <typename T>
void exchange_results(device_buffer<T> &res, device_buffer<int64_t> &res_I,
                      device_buffer<float> &res_D,
                      const raft::comms::comms_t &comm, int part_rank,
                      std::set<int> idxRanks, cudaStream_t stream,
                      size_t cur_batch_size, int k, int n_outputs,
                      int local_parts_completed);

void perform_local_knn(int64_t *res_I, float *res_D,
                       std::vector<Matrix::floatData_t *> &idx_data,
                       Matrix::PartDescriptor &idx_desc,
                       std::vector<Matrix::RankSizePair *> &local_idx_parts,
                       std::vector<size_t> &start_indices, cudaStream_t stream,
                       cudaStream_t *internal_streams, int n_internal_streams,
                       std::shared_ptr<deviceAllocator> allocator,
                       size_t cur_batch_size, int k, float *cur_query_ptr,
                       bool rowMajorIndex, bool rowMajorQuery);

template <typename T>
void perform_local_operation(T *out, int64_t *knn_indices, T *labels,
                             size_t cur_batch_size, int k, int n_outputs,
                             raft::handle_t &h, bool probas_only = false,
                             std::vector<float *> *probas = nullptr,
                             std::vector<int *> *uniq_labels = nullptr,
                             std::vector<int> *n_unique = nullptr);

template <typename T>
void copy_outputs(T *out, int64_t *knn_indices,
                  std::vector<std::vector<T *>> &y, size_t cur_batch_size,
                  int k, int n_outputs, int n_features, int my_rank,
                  std::vector<Matrix::RankSizePair *> &idxPartsToRanks,
                  std::shared_ptr<deviceAllocator> alloc, cudaStream_t stream);
};  // namespace knn_common
};  // namespace opg
};  // namespace KNN
};  // namespace ML