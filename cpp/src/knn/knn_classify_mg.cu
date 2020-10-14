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

void knn_classify(ML::cumlHandle &handle, std::vector<Matrix::Data<int> *> *out,
                  std::vector<Matrix::Data<int64_t> *> *out_I,
                  std::vector<Matrix::floatData_t *> *out_D,
                  std::vector<std::vector<float *>> *probas,
                  std::vector<Matrix::floatData_t *> &idx_data,
                  Matrix::PartDescriptor &idx_desc,
                  std::vector<Matrix::floatData_t *> &query_data,
                  Matrix::PartDescriptor &query_desc,
                  std::vector<std::vector<int *>> &y,
                  std::vector<int *> &uniq_labels, std::vector<int> &n_unique,
                  bool rowMajorIndex, bool rowMajorQuery, bool probas_only,
                  int k, size_t batch_size, bool verbose) {
  int n_outputs = n_unique.size();

  opg_knn(handle, out, out_I, out_D, idx_data, idx_desc, query_data, query_desc,
          y, rowMajorIndex, rowMajorQuery, k, n_outputs, batch_size, verbose,
          probas, &uniq_labels, &n_unique, probas_only);
}
};  // namespace opg
};  // namespace KNN
};  // namespace ML
