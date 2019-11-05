/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cuda_utils.h>
#include <gtest/gtest.h>
#include <linalg/cublas_wrappers.h>
#include <random/make_blobs.h>
#include <test_utils.h>
#include <common/cumlHandle.hpp>
#include <cuml/cuml.hpp>
#include <decisiontree/batched-levelalgo/builder.cuh>

namespace ML {
namespace DecisionTree {

struct DtTestParams {
  int M, N, nclasses, max_depth, nbins;
  float min_gain;
  unsigned long long seed;
};

::std::ostream& operator<<(::std::ostream& os, const DtTestParams& dims) {
  return os;
}

template <typename T, typename L = int, typename I = int>
class DtBaseTest : public ::testing::TestWithParam<DtTestParams> {
 protected:
  void SetUp() {
    inparams = ::testing::TestWithParam<DtTestParams>::GetParam();
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.setStream(stream);
    set_tree_params(params, inparams.max_depth, 1 << inparams.max_depth, 1.f,
                    inparams.nbins, SPLIT_ALGO::GLOBAL_QUANTILE, inparams.nbins,
                    inparams.min_gain, false, CRITERION::GINI, false, false, 32,
                    10, 4, 0);
    prepareDataset();
    auto impl = handle.getImpl();
    grow_tree<T, L, I>(impl.getDeviceAllocator(), impl.getHostAllocator(), data,
                       inparams.N, inparams.M, labels, quantiles, rowids,
                       colids, inparams.M, inparams.nclasses, params, stream,
                       sparsetree);
  }

  void TearDown() {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto allocator = handle.getImpl().getDeviceAllocator();
    allocator->deallocate(data, sizeof(T) * inparams.M * inparams.N, stream);
    allocator->deallocate(labels, sizeof(L) * inparams.M, stream);
    ///@todo: deallocate rowids and colids
    ///@todo: deallocate quantiles
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  cudaStream_t stream;
  cumlHandle handle;
  T *data, *quantiles;
  L* labels;
  I *rowids, *colids;
  DecisionTreeParams params;
  DtTestParams inparams;
  std::vector<SparseTreeNode<T, L>> sparsetree;

 private:
  ///@todo: support regression
  void prepareDataset() {
    auto allocator = handle.getImpl().getDeviceAllocator();
    auto cublas = handle.getImpl().getCublasHandle();
    data = (T*)allocator->allocate(sizeof(T) * inparams.M * inparams.N, stream);
    labels = (L*)allocator->allocate(sizeof(L) * inparams.M, stream);
    auto* tmp =
      (T*)allocator->allocate(sizeof(T) * inparams.M * inparams.N, stream);
    MLCommon::Random::make_blobs<T>(
      tmp, labels, inparams.M, inparams.N, inparams.nclasses, allocator, stream,
      nullptr, nullptr, T(1.0), false, T(10.0), T(-10.0), inparams.seed);
    auto alpha = T(1.0), beta = T(0.0);
    CUBLAS_CHECK(MLCommon::LinAlg::cublasgeam(
      cublas, CUBLAS_OP_T, CUBLAS_OP_N, inparams.M, inparams.N, &alpha, tmp,
      inparams.N, &beta, tmp, inparams.M, data, inparams.M, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    allocator->deallocate(tmp, sizeof(T) * inparams.M * inparams.N, stream);
    ///@todo: allocate and populate rowids and colids
    ///@todo: allocate and populate quantiles
  }
};  // class DtBaseTest

const std::vector<DtTestParams> all = {
  {1024, 4, 2, 8, 16, 0.00001f, 12345ULL},
  {1024, 4, 2, 8, 16, 0.00001f, 12345ULL},
};
typedef DtBaseTest<float> DtTestF;
///@todo
TEST_P(DtTestF, Test) {}
INSTANTIATE_TEST_CASE_P(BatchedLevelAlgo, DtTestF, ::testing::ValuesIn(all));

}  // end namespace DecisionTree
}  // end namespace ML
