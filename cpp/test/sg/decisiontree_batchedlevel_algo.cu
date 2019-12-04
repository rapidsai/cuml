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
#include <decisiontree/quantile/quantile.h>
#include <gtest/gtest.h>
#include <linalg/cublas_wrappers.h>
#include <random/make_blobs.h>
#include <random/make_regression.h>
#include <test_utils.h>
#include <common/cumlHandle.hpp>
#include <common/iota.cuh>
#include <cuml/cuml.hpp>
#include <decisiontree/batched-levelalgo/builder.cuh>
#include <memory>

namespace ML {
namespace DecisionTree {

struct DtTestParams {
  int M, N, nclasses, max_depth, nbins;
  float min_gain;
  CRITERION splitType;
  unsigned long long seed;
};

::std::ostream& operator<<(::std::ostream& os, const DtTestParams& dims) {
  return os;
}

template <typename T, typename L, typename I = int>
class DtBaseTest : public ::testing::TestWithParam<DtTestParams> {
 protected:
  void SetUp() {
    inparams = ::testing::TestWithParam<DtTestParams>::GetParam();
    handle.reset(new cumlHandle);
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle->setStream(stream);
    set_tree_params(params, inparams.max_depth, 1 << inparams.max_depth, 1.f,
                    inparams.nbins, SPLIT_ALGO::GLOBAL_QUANTILE, inparams.nbins,
                    inparams.min_gain, false, inparams.splitType, false, false,
                    128, 10, 4, 0);
    auto allocator = handle->getImpl().getDeviceAllocator();
    data = (T*)allocator->allocate(sizeof(T) * inparams.M * inparams.N, stream);
    labels = (L*)allocator->allocate(sizeof(L) * inparams.M, stream);
    auto* tmp =
      (T*)allocator->allocate(sizeof(T) * inparams.M * inparams.N, stream);
    prepareDataset(tmp);
    auto alpha = T(1.0), beta = T(0.0);
    auto cublas = handle->getImpl().getCublasHandle();
    CUBLAS_CHECK(MLCommon::LinAlg::cublasgeam(
      cublas, CUBLAS_OP_T, CUBLAS_OP_N, inparams.M, inparams.N, &alpha, tmp,
      inparams.N, &beta, tmp, inparams.M, data, inparams.M, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    allocator->deallocate(tmp, sizeof(T) * inparams.M * inparams.N, stream);
    rowids = (I*)allocator->allocate(sizeof(I) * inparams.M, stream);
    MLCommon::iota(rowids, 0, 1, inparams.M, stream);
    colids = (I*)allocator->allocate(sizeof(I) * inparams.N, stream);
    MLCommon::iota(colids, 0, 1, inparams.N, stream);
    quantiles =
      (T*)allocator->allocate(sizeof(T) * inparams.nbins * inparams.N, stream);
    preprocess_quantile<T>((const T*)data, (const unsigned*)rowids, inparams.M,
                           inparams.N, inparams.M, inparams.nbins, (T*)nullptr,
                           quantiles, (T*)nullptr, allocator, stream);
  }

  void TearDown() {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto allocator = handle->getImpl().getDeviceAllocator();
    allocator->deallocate(data, sizeof(T) * inparams.M * inparams.N, stream);
    allocator->deallocate(labels, sizeof(L) * inparams.M, stream);
    allocator->deallocate(rowids, sizeof(int) * inparams.M, stream);
    allocator->deallocate(colids, sizeof(int) * inparams.N, stream);
    allocator->deallocate(quantiles, sizeof(T) * inparams.nbins * inparams.N,
                          stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    handle.reset();
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  cudaStream_t stream;
  std::shared_ptr<cumlHandle> handle;
  T *data, *quantiles;
  L* labels;
  I *rowids, *colids;
  DecisionTreeParams params;
  DtTestParams inparams;
  std::vector<SparseTreeNode<T, L>> sparsetree;

  virtual void prepareDataset(T* tmp) = 0;
};  // class DtBaseTest

const std::vector<DtTestParams> allC = {
  {1024, 4, 2, 8, 16, 0.00001f, CRITERION::GINI, 12345ULL},
  {1024, 4, 2, 8, 16, 0.00001f, CRITERION::GINI, 12345ULL},
  {1024, 4, 2, 8, 16, 0.00001f, CRITERION::ENTROPY, 12345ULL},
  {1024, 4, 2, 8, 16, 0.00001f, CRITERION::ENTROPY, 12345ULL},
};
template <typename T>
class DtClassifierTest : public DtBaseTest<T, int> {
 protected:
  void prepareDataset(T* tmp) override {
    auto allocator = this->handle->getImpl().getDeviceAllocator();
    auto inparams = this->inparams;
    MLCommon::Random::make_blobs<T>(tmp, this->labels, inparams.M, inparams.N,
                                    inparams.nclasses, allocator, this->stream,
                                    nullptr, nullptr, T(1.0), false, T(10.0),
                                    T(-10.0), inparams.seed);
  }
};  // class DtClassifierTest
typedef DtClassifierTest<float> DtClsTestF;
///@todo: add checks
TEST_P(DtClsTestF, Test) {
  auto& impl = handle->getImpl();
  int num_leaves, depth;
  grow_tree<float, int, int>(impl.getDeviceAllocator(), impl.getHostAllocator(),
                             data, inparams.N, inparams.M, labels, quantiles,
                             rowids, colids, inparams.M, inparams.nclasses,
                             params, stream, sparsetree, num_leaves, depth);
  // dataset is such that it makes the tree reach all the way to max_depth
  ASSERT_EQ(depth, inparams.max_depth);
}
INSTANTIATE_TEST_CASE_P(BatchedLevelAlgo, DtClsTestF,
                        ::testing::ValuesIn(allC));

const std::vector<DtTestParams> allR = {
  {1024, 4, 2, 8, 16, 0.00001f, CRITERION::MSE, 12345ULL},
  {1024, 4, 2, 8, 16, 0.00001f, CRITERION::MSE, 12345ULL},
};
template <typename T>
class DtRegressorTest : public DtBaseTest<T, T> {
 protected:
  void prepareDataset(T* tmp) override {
    auto allocator = this->handle->getImpl().getDeviceAllocator();
    auto cublas = this->handle->getImpl().getCublasHandle();
    auto cusolver = this->handle->getImpl().getcusolverDnHandle();
    auto inparams = this->inparams;
    MLCommon::Random::make_regression<T>(
      tmp, this->labels, inparams.M, inparams.N, inparams.N, cublas, cusolver,
      allocator, this->stream, nullptr, 1, T(1.0), -1, T(0.5), T(0.0), false,
      inparams.seed);
  }
};  // class DtRegressorTest
typedef DtRegressorTest<float> DtRegTestF;
///@todo: add checks
TEST_P(DtRegTestF, Test) {
  auto& impl = handle->getImpl();
  int num_leaves, depth;
  grow_tree<float, int>(impl.getDeviceAllocator(), impl.getHostAllocator(),
                        data, inparams.N, inparams.M, labels, quantiles, rowids,
                        colids, inparams.M, 0, params, stream, sparsetree,
                        num_leaves, depth);
  ASSERT_EQ(sparsetree.size(), 1);
  ASSERT_EQ(num_leaves, 1);
  ASSERT_EQ(depth, 0);
}
INSTANTIATE_TEST_CASE_P(BatchedLevelAlgo, DtRegTestF,
                        ::testing::ValuesIn(allR));

}  // namespace DecisionTree
}  // end namespace ML
