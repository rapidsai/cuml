/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>
#include <vector>

#include <cuml/cluster/linkage.hpp>
#include <cuml/common/cuml_allocator.hpp>
#include <cuml/cuml.hpp>
#include <cuml/datasets/make_blobs.hpp>
#include <cuml/metrics/metrics.hpp>

#include <raft/linalg/cublas_wrappers.h>
#include <raft/linalg/distance_type.h>
#include <raft/linalg/transpose.h>

#include <test_utils.h>

#include <common/device_buffer.hpp>
#include <cuml/common/logger.hpp>

namespace ML {

using namespace MLCommon;
using namespace Datasets;
using namespace Metrics;
using namespace std;

template <typename T, typename IdxT>
struct LinkageInputs {
  IdxT n_row;
  IdxT n_col;
  IdxT n_centers;
  T cluster_std;
  bool use_knn;
  unsigned long long int seed;
};

template <typename T, typename IdxT>
::std::ostream &operator<<(::std::ostream &os,
                           const LinkageInputs<T, IdxT> &dims) {
  return os;
}

template <typename T, typename IdxT>
class LinkageTest : public ::testing::TestWithParam<LinkageInputs<T, IdxT>> {
 protected:
  void basicTest() {
    raft::handle_t handle;

    params = ::testing::TestWithParam<LinkageInputs<T, IdxT>>::GetParam();

    device_buffer<T> out(handle.get_device_allocator(), handle.get_stream(),
                         params.n_row * params.n_col);
    device_buffer<IdxT> l(handle.get_device_allocator(), handle.get_stream(),
                          params.n_row);

    make_blobs(handle, out.data(), l.data(), params.n_row, params.n_col,
               params.n_centers, true, nullptr, nullptr, params.cluster_std,
               true, -10.0f, 10.0f, params.seed);

    raft::allocate(labels, params.n_row);
    raft::allocate(labels_ref, params.n_row);

    raft::copy(labels_ref, l.data(), params.n_row, handle.get_stream());

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    ML::linkage_output<IdxT, T> out_arrs;

    device_buffer<IdxT> out_children(handle.get_device_allocator(),
                                     handle.get_stream(),
                                     (params.n_row - 1) * 2);
    out_arrs.children = out_children.data();
    out_arrs.labels = l.data();

    LinkageDistance dist_type =
      params.use_knn ? LinkageDistance::KNN_GRAPH : LinkageDistance::PAIRWISE;

    CUML_LOG_INFO("Dist_type: %d", dist_type);

    ML::single_linkage(handle, out.data(), params.n_row, params.n_col,
                       raft::distance::DistanceType::EucExpandedL2, dist_type,
                       &out_arrs);

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    //    score = adjusted_rand_index(handle, labels_ref, labels, params.n_row);
    //
    //    if (score < 1.0) {
    //      auto str = raft::arr2Str(labels_ref, params.n_row, "labels_ref",
    //                               handle.get_stream());
    //      CUML_LOG_DEBUG("y: %s", str.c_str());
    //      str = raft::arr2Str(labels, params.n_row, "labels", handle.get_stream());
    //      CUML_LOG_DEBUG("y_hat: %s", str.c_str());
    //      CUML_LOG_DEBUG("Score = %lf", score);
    //    }
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {
    CUDA_CHECK(cudaFree(labels));
    CUDA_CHECK(cudaFree(labels_ref));
  }

 protected:
  LinkageInputs<T, IdxT> params;
  IdxT *labels, *labels_ref;

  double score;
};

const std::vector<LinkageInputs<float, int>> inputsf2 = {
  {500, 16, 5, 0.01, true, 1234ULL},
  //  {1000, 1000, 10, 0.01, false, 1234ULL},
  //  {20000, 10000, 10, 0.01, false, 1234ULL},
  //  {20000, 100, 5000, 0.01, false, 1234ULL}
};

typedef LinkageTest<float, int> LinkageTestF_Int;
TEST_P(LinkageTestF_Int, Result) { ASSERT_TRUE(score == 1.0); }

INSTANTIATE_TEST_CASE_P(LinkageTests, LinkageTestF_Int,
                        ::testing::ValuesIn(inputsf2));
}  // end namespace ML
