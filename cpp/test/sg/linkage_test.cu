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

  std::vector<T> data;

  std::vector<IdxT> expected_labels;

  int n_clusters;

  bool use_knn;
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

    device_buffer<T> data(handle.get_device_allocator(), handle.get_stream(),
                          params.n_row * params.n_col);
    device_buffer<IdxT> l(handle.get_device_allocator(), handle.get_stream(),
                          params.n_row);

    raft::allocate(labels, params.n_row);
    raft::allocate(labels_ref, params.n_row);

    raft::copy(data.data(), params.data.data(), data.size(),
               handle.get_stream());
    raft::copy(labels_ref, params.expected_labels.data(), params.n_row,
               handle.get_stream());

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

    ML::single_linkage(handle, data.data(), params.n_row, params.n_col,
                       raft::distance::DistanceType::L2Expanded, dist_type,
                       &out_arrs, 1);

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
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

/**
 *
 * mst: [[3.         1.         0.47451804]
 [8.         6.         0.47528009]
 [1.         7.         0.47946792]
 [0.         9.         0.53807401]
 [7.         4.         0.54156098]
 [9.         3.         0.59827752]
 [4.         2.         0.6234549 ]
 [2.         8.         0.62997891]
 [6.         5.         0.63858579]]

 * [[0.21390334, 0.50261639, 0.91036676, 0.59166485, 0.71162682],
       [0.10248392, 0.77782677, 0.43772379, 0.4035871 , 0.32827965],
       [0.47544681, 0.59862974, 0.12319357, 0.06239463, 0.28200272],
       [0.1345717 , 0.50498218, 0.5113505 , 0.16233086, 0.62165332],
       [0.42281548, 0.933117  , 0.41386077, 0.23264562, 0.73325968],
       [0.37537541, 0.70719873, 0.14522645, 0.73279625, 0.9126674 ],
       [0.84854131, 0.28890216, 0.85267903, 0.74703138, 0.83842071],
       [0.34942792, 0.27864171, 0.70911132, 0.21338564, 0.32035554],
       [0.73788331, 0.46926692, 0.57570162, 0.42559178, 0.87120209],
       [0.22734951, 0.01847905, 0.75549396, 0.76166195, 0.66613745]]

children [[ 3  1]
 [ 8  6]
 [10  7]
 [ 0  9]
 [12  4]
 [13 14]
 [15  2]
 [16 11]
 [17  5]]

 nodes: [-18]
nodes: [-17, -5]
nodes: [-16, -5, -11]
nodes: [-15, -5, -11, -2]
nodes: [-14, -13, -11, -2, -5]
labels: [1 0 3 0 0 4 2 0 2 1]


 */

const std::vector<LinkageInputs<float, int>> inputsf2 = {
  {10,
   5,
   {0.21390334, 0.50261639, 0.91036676, 0.59166485, 0.71162682, 0.10248392,
    0.77782677, 0.43772379, 0.4035871,  0.3282796,  0.47544681, 0.59862974,
    0.12319357, 0.06239463, 0.28200272, 0.1345717,  0.50498218, 0.5113505,
    0.16233086, 0.62165332, 0.42281548, 0.933117,   0.41386077, 0.23264562,
    0.73325968, 0.37537541, 0.70719873, 0.14522645, 0.73279625, 0.9126674,
    0.84854131, 0.28890216, 0.85267903, 0.74703138, 0.83842071, 0.34942792,
    0.27864171, 0.70911132, 0.21338564, 0.32035554, 0.73788331, 0.46926692,
    0.57570162, 0.42559178, 0.87120209, 0.22734951, 0.01847905, 0.75549396,
    0.76166195, 0.66613745},
   {1, 0, 3, 0, 0, 4, 2, 0, 2, 1},
   5,
   true},
};

typedef LinkageTest<float, int> LinkageTestF_Int;
TEST_P(LinkageTestF_Int, Result) { ASSERT_TRUE(score == 1.0); }

INSTANTIATE_TEST_CASE_P(LinkageTests, LinkageTestF_Int,
                        ::testing::ValuesIn(inputsf2));
}  // end namespace ML
