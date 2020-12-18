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

#include "knn_opg_common.cuh"

namespace ML {
namespace KNN {
namespace opg {

using namespace knn_common;

void knn_regress(raft::handle_t &handle,
                 std::vector<Matrix::Data<float> *> *out,
                 std::vector<Matrix::Data<int64_t> *> *out_I,
                 std::vector<Matrix::floatData_t *> *out_D,
                 std::vector<Matrix::floatData_t *> &idx_data,
                 Matrix::PartDescriptor &idx_desc,
                 std::vector<Matrix::floatData_t *> &query_data,
                 Matrix::PartDescriptor &query_desc,
                 std::vector<std::vector<float *>> &y, bool rowMajorIndex,
                 bool rowMajorQuery, int k, int n_outputs, size_t batch_size,
                 bool verbose) {
  opg_knn_param params;
  params.knn_op = knn_operation::regression;
  params.out.f = out;
  params.out_I = out_I;
  params.out_D = out_D;
  params.idx_data = &idx_data;
  params.idx_desc = &idx_desc;
  params.query_data = &query_data;
  params.query_desc = &query_desc;
  params.y.f = &y;
  params.rowMajorIndex = rowMajorIndex;
  params.rowMajorQuery = rowMajorQuery;
  params.k = k;
  params.n_outputs = n_outputs;
  params.batch_size = batch_size;
  params.verbose = verbose;

  cuda_utils cutils(handle);
  opg_knn(params, cutils);
}
};  // namespace opg
};  // namespace KNN
};  // namespace ML
