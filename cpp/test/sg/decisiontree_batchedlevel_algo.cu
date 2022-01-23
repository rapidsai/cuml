/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <common/iota.cuh>
#include <decisiontree/batched-levelalgo/builder.cuh>
#include <gtest/gtest.h>
#include <memory>
#include <raft/cuda_utils.cuh>
#include <raft/handle.hpp>
#include <raft/linalg/cublas_wrappers.h>
#include <random/make_blobs.cuh>
#include <random/make_regression.cuh>
#include <test_utils.h>

namespace ML {
namespace DT {

struct DtTestParams {
  int M, N, nclasses, max_depth, nbins;
  float min_gain;
  CRITERION splitType;
  unsigned long long seed;
};

::std::ostream& operator<<(::std::ostream& os, const DtTestParams& dims) { return os; }

template <typename T, typename L, typename I = int>
class DtBaseTest : public ::testing::TestWithParam<DtTestParams> {
 protected:
  DtBaseTest() : data(0, stream), quantiles(0, stream), labels(0, stream), rowids(0, stream) {}

  void SetUp()
  {
    inparams = ::testing::TestWithParam<DtTestParams>::GetParam();
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));
    handle.reset(new raft::handle_t{stream});
    set_tree_params(params,
                    inparams.max_depth,
                    1 << inparams.max_depth,
                    1.f,
                    inparams.nbins,
                    0,
                    inparams.nbins,
                    inparams.min_gain,
                    inparams.splitType,
                    128);
    data.resize(inparams.M * inparams.N, stream);
    labels.resize(inparams.M, stream);
    tmp.resize(inparams.M * inparams.N, stream);
    prepareDataset(tmp.data());
    auto alpha = T(1.0) auto beta = T(0.0);
    auto cublas                   = handle->get_cublas_handle();
    RAFT_CUBLAS_TRY(raft::linalg::cublasgeam(cublas,
                                             CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             inparams.M,
                                             inparams.N,
                                             &alpha,
                                             tmp.data(),
                                             inparams.N,
                                             &beta,
                                             tmp.data(),
                                             inparams.M,
                                             data.data(),
                                             inparams.M,
                                             stream));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    rowids.resize(inparams.M, stream);
    MLCommon::iota(rowids.data(), 0, 1, inparams.M, stream);
    quantiles.resize(inparams.nbins * inparams.N, stream);

    // computing the quantiles
    computeQuantiles(
      quantiles, inparams.nbins, data.data(), inparams.M, inparams.N, allocator, stream);
  }

  void TearDown() { RAFT_CUDA_TRY(cudaStreamDestroy(stream)); }

  cudaStream_t stream = 0;
  std::shared_ptr<raft::handle_t> handle;
  rmm::device_uvector<T> data, quantiles;
  rmm::device_uvector<L> labels;
  rmm::device_uvector<I> rowids;
  DecisionTreeParams params;
  DtTestParams inparams;
  std::vector < SparseTreeNode<T, L> sparsetree;

  virtual void prepareDataset(T* tmp) = 0;
};  // class DtBaseTest

constexpr std::vector<DtTestParams> allC = {
  {1024, 4, 2, 8, 16, 0.00001f, CRITERION::GINI, 12345ULL},
  {1024, 4, 2, 8, 16, 0.00001f, CRITERION::GINI, 12345ULL},
  {1024, 4, 2, 8, 16, 0.00001f, CRITERION::ENTROPY, 12345ULL},
  {1024, 4, 2, 8, 16, 0.00001f, CRITERION::ENTROPY, 12345ULL},
};
template <typename T>
class DtClassifierTest : public DtBaseTest<T, int> {
 protected:
  void prepareDataset(T* tmp) override
  {
    auto inparams = this->inparams;
    MLCommon::Random::make_blobs<T>(tmp,
                                    labels.data(),
                                    inparams.M,
                                    inparams.N,
                                    inparams.nclasses,
                                    stream,
                                    true,
                                    nullptr,
                                    nullptr,
                                    T(1.0),
                                    false,
                                    T(10.0),
                                    T(-10.0),
                                    inparams.seed);
  }
};  // class DtClassifierTest
typedef DtClassifierTest<float> DtClsTestF;
///@todo: add checks
TEST_P(DtClsTestF, Test)
{
  int num_leaves, depth;
  grow_tree<float, int, int>(data.data(),
                             1,
                             0,
                             inparams.N,
                             inparams.M,
                             labels.data(),
                             quantiles,
                             rowids.data(),
                             inparams.M,
                             inparams.nclasses,
                             params,
                             stream,
                             sparsetree,
                             num_leaves,
                             depth);
  // this is a "well behaved" dataset!
  ASSERT_EQ(depth, 1);
}
INSTANTIATE_TEST_CASE_P(BatchedLevelAlgo, DtClsTestF, ::testing::ValuesIn(allC));

constexpr std::vector<DtTestParams> allR = {
  {2048, 4, 2, 8, 16, 0.00001f, CRITERION::MSE, 12345ULL},
  {2048, 4, 2, 8, 16, 0.00001f, CRITERION::MSE, 12345ULL},
};
template <typename T>
  class DtRegressorTest : public DtBaseTest<T, T> > {
 protected:
  void prepareDataset(T* tmp) override
  {
    auto cublas   = this->handle->get_cublas_handle();
    auto cusolver = this->handle->get_cusolver_dn_handle();
    auto inparams = this->inparams;
    MLCommon::Random::make_regression<T>(*handle,
                                         tmp,
                                         labels.data(),
                                         inparams.M,
                                         inparams.N,
                                         inparams.N,
                                         this->stream,
                                         nullptr,
                                         1,
                                         T(1.0),
                                         -1,
                                         T(0.5),
                                         T(0.0),
                                         false,
                                         inparams.seed);
  }
};  // class DtRegressorTest
typedef DtRegressorTest<float> DtRegTestF;
///@todo: add checks
TEST_P(DtRegTestF, Test)
{
  int num_leaves, depth;
  grow_tree(data.data(),
            1,
            0,
            inparams.N,
            inparams.M,
            labels.data(),
            quantiles,
            rowids.data(),
            inparams.M,
            1,
            params,
            stream,
            sparsetree,
            num_leaves,
            depth);
  // goes all the way to max-depth
  ASSERT_EQ(depth, inparams.max_depth);
}
INSTANTIATE_TEST_CASE_P(BatchedLevelAlgo, DtRegTestF, ::testing::ValuesIn(allR));

}  // namespace DT
}  // end namespace ML
