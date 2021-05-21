/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include "hdbscan_inputs.hpp"

#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>
#include <vector>

#include <hdbscan/detail/utils.h>
#include <cuml/cluster/hdbscan.hpp>
#include <hdbscan/detail/condense.cuh>
#include <hdbscan/detail/extract.cuh>

#include <metrics/adjusted_rand_index.cuh>

#include <raft/sparse/hierarchy/detail/agglomerative.cuh>

#include <raft/linalg/distance_type.h>
#include <raft/linalg/transpose.h>
#include <raft/sparse/op/sort.h>
#include <raft/mr/device/allocator.hpp>
#include <raft/sparse/coo.cuh>
#include <rmm/device_uvector.hpp>

#include "../prims/test_utils.h"

namespace ML {
namespace HDBSCAN {

using namespace std;

template <typename T, typename IdxT>
::std::ostream& operator<<(::std::ostream& os,
                           const HDBSCANInputs<T, IdxT>& dims) {
  return os;
}

template <typename T, typename IdxT>
class HDBSCANTest : public ::testing::TestWithParam<HDBSCANInputs<T, IdxT>> {
 protected:
  void basicTest() {
    raft::handle_t handle;

    params = ::testing::TestWithParam<HDBSCANInputs<T, IdxT>>::GetParam();

    rmm::device_uvector<T> data(params.n_row * params.n_col,
                                handle.get_stream());

    // Allocate result labels and expected labels on device
    raft::allocate(labels_ref, params.n_row);

    raft::copy(data.data(), params.data.data(), data.size(),
               handle.get_stream());
    raft::copy(labels_ref, params.expected_labels.data(), params.n_row,
               handle.get_stream());

    rmm::device_uvector<IdxT> out_children(params.n_row * 2,
                                           handle.get_stream());
    rmm::device_uvector<T> out_deltas(params.n_row, handle.get_stream());

    rmm::device_uvector<IdxT> out_sizes(params.n_row * 2, handle.get_stream());

    rmm::device_uvector<IdxT> out_labels(params.n_row, handle.get_stream());

    rmm::device_uvector<IdxT> mst_src(params.n_row - 1, handle.get_stream());
    rmm::device_uvector<IdxT> mst_dst(params.n_row - 1, handle.get_stream());
    rmm::device_uvector<T> mst_weights(params.n_row - 1, handle.get_stream());

    rmm::device_uvector<T> out_probabilities(params.n_row, handle.get_stream());

    Logger::get().setLevel(CUML_LEVEL_DEBUG);

    HDBSCAN::Common::hdbscan_output<IdxT, T> out(
      handle, params.n_row, out_labels.data(), out_probabilities.data(),
      out_children.data(), out_sizes.data(), out_deltas.data(), mst_src.data(),
      mst_dst.data(), mst_weights.data());

    HDBSCAN::Common::HDBSCANParams hdbscan_params;
    hdbscan_params.k = params.k;
    hdbscan_params.min_cluster_size = params.min_cluster_size;
    hdbscan_params.min_samples = params.min_pts;

    hdbscan(handle, data.data(), params.n_row, params.n_col,
            raft::distance::DistanceType::L2SqrtExpanded, hdbscan_params, out);

    raft::print_device_vector("outlabels", out.get_labels(), params.n_row,
                              std::cout);

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    score = MLCommon::Metrics::compute_adjusted_rand_index(
      out.get_labels(), labels_ref, params.n_row, handle.get_device_allocator(),
      handle.get_stream());
  }

  void SetUp() override { basicTest(); }

  void TearDown() override { CUDA_CHECK(cudaFree(labels_ref)); }

 protected:
  HDBSCANInputs<T, IdxT> params;
  IdxT* labels_ref;
  int k;

  double score;
};

typedef HDBSCANTest<float, int> HDBSCANTestF_Int;
TEST_P(HDBSCANTestF_Int, Result) { EXPECT_TRUE(score == 1.0); }

INSTANTIATE_TEST_CASE_P(HDBSCANTest, HDBSCANTestF_Int,
                        ::testing::ValuesIn(hdbscan_inputsf2));

template <typename T, typename IdxT>
class ClusterCondensingTest
  : public ::testing::TestWithParam<ClusterCondensingInputs<T, IdxT>> {
 protected:
  void basicTest() {
    raft::handle_t handle;

    params =
      ::testing::TestWithParam<ClusterCondensingInputs<T, IdxT>>::GetParam();

    rmm::device_uvector<IdxT> mst_src(params.n_row - 1, handle.get_stream());
    rmm::device_uvector<IdxT> mst_dst(params.n_row - 1, handle.get_stream());
    rmm::device_uvector<T> mst_data(params.n_row - 1, handle.get_stream());

    raft::copy(mst_src.data(), params.mst_src.data(), params.mst_src.size(),
               handle.get_stream());

    raft::copy(mst_dst.data(), params.mst_dst.data(), params.mst_dst.size(),
               handle.get_stream());

    raft::copy(mst_data.data(), params.mst_data.data(), params.mst_data.size(),
               handle.get_stream());

    rmm::device_uvector<IdxT> expected_device(params.expected.size(),
                                              handle.get_stream());
    raft::copy(expected_device.data(), params.expected.data(),
               params.expected.size(), handle.get_stream());

    rmm::device_uvector<IdxT> out_children(params.n_row * 2,
                                           handle.get_stream());

    rmm::device_uvector<IdxT> out_size(params.n_row, handle.get_stream());

    rmm::device_uvector<T> out_delta(params.n_row, handle.get_stream());

    Logger::get().setLevel(CUML_LEVEL_DEBUG);

    raft::sparse::op::coo_sort_by_weight(mst_src.data(), mst_dst.data(),
                                         mst_data.data(), (IdxT)mst_src.size(),
                                         handle.get_stream());

    /**
     * Build dendrogram of MST
     */
    raft::hierarchy::detail::build_dendrogram_host(
      handle, mst_src.data(), mst_dst.data(), mst_data.data(), params.n_row - 1,
      out_children.data(), out_delta.data(), out_size.data());

    raft::print_device_vector("children", out_children.data(),
                              out_children.size(), std::cout);
    raft::print_device_vector("delta", out_delta.data(), out_delta.size(),
                              std::cout);
    raft::print_device_vector("size", out_size.data(), out_delta.size(),
                              std::cout);

    /**
     * Condense Hierarchy
     */
    HDBSCAN::Common::CondensedHierarchy<IdxT, T> condensed_tree(handle,
                                                                params.n_row);
    HDBSCAN::detail::Condense::build_condensed_hierarchy(
      handle, out_children.data(), out_delta.data(), out_size.data(),
      params.min_cluster_size, params.n_row, condensed_tree);

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    raft::print_device_vector("condensed parents", condensed_tree.get_parents(),
                              condensed_tree.get_n_edges(), std::cout);
    raft::print_device_vector("condensed children",
                              condensed_tree.get_children(),
                              condensed_tree.get_n_edges(), std::cout);
    raft::print_device_vector("condensed size", condensed_tree.get_sizes(),
                              condensed_tree.get_n_edges(), std::cout);

    rmm::device_uvector<IdxT> labels(params.n_row, handle.get_stream());
    rmm::device_uvector<T> stabilities(condensed_tree.get_n_clusters(),
                                       handle.get_stream());
    rmm::device_uvector<T> probabilities(params.n_row, handle.get_stream());

    HDBSCAN::detail::Extract::extract_clusters(
      handle, condensed_tree, params.n_row, labels.data(), stabilities.data(),
      probabilities.data(), HDBSCAN::Common::CLUSTER_SELECTION_METHOD::EOM,
      true);

    CUML_LOG_DEBUG("Evaluating results");
    if (params.expected.size() == params.n_row) {
      raft::print_device_vector("labels", labels.data(), labels.size(),
                                std::cout);
      score = MLCommon::Metrics::compute_adjusted_rand_index(
        labels.data(), expected_device.data(), params.n_row,
        handle.get_device_allocator(), handle.get_stream());
    } else {
      score = 1.0;
    }
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  ClusterCondensingInputs<T, IdxT> params;
  int k;

  double score;
};

typedef ClusterCondensingTest<float, int> ClusterCondensingTestF_Int;
TEST_P(ClusterCondensingTestF_Int, Result) { EXPECT_TRUE(score == 1.0); }

INSTANTIATE_TEST_CASE_P(ClusterCondensingTest, ClusterCondensingTestF_Int,
                        ::testing::ValuesIn(cluster_condensing_inputs));

template <typename T, typename IdxT>
class ClusterSelectionTest
  : public ::testing::TestWithParam<ClusterSelectionInputs<T, IdxT>> {
 protected:
  void basicTest() {
    raft::handle_t handle;

    params =
      ::testing::TestWithParam<ClusterSelectionInputs<T, IdxT>>::GetParam();

    Logger::get().setLevel(CUML_LEVEL_DEBUG);

    rmm::device_uvector<IdxT> condensed_parents(params.condensed_parents.size(),
                                                handle.get_stream());
    rmm::device_uvector<IdxT> condensed_children(
      params.condensed_children.size(), handle.get_stream());
    rmm::device_uvector<T> condensed_lambdas(params.condensed_lambdas.size(),
                                             handle.get_stream());
    rmm::device_uvector<IdxT> condensed_sizes(params.condensed_sizes.size(),
                                              handle.get_stream());

    // outputs
    rmm::device_uvector<T> stabilities(params.n_row, handle.get_stream());
    rmm::device_uvector<T> probabilities(params.n_row, handle.get_stream());
    rmm::device_uvector<IdxT> labels(params.n_row, handle.get_stream());

    raft::copy(condensed_parents.data(), params.condensed_parents.data(),
               condensed_parents.size(), handle.get_stream());

    raft::copy(condensed_children.data(), params.condensed_children.data(),
               condensed_children.size(), handle.get_stream());

    raft::copy(condensed_lambdas.data(), params.condensed_lambdas.data(),
               condensed_lambdas.size(), handle.get_stream());

    raft::copy(condensed_sizes.data(), params.condensed_sizes.data(),
               condensed_sizes.size(), handle.get_stream());

    ML::HDBSCAN::Common::CondensedHierarchy<IdxT, T> condensed_tree(
      handle, params.n_row, params.condensed_parents.size(),
      condensed_parents.data(), condensed_children.data(),
      condensed_lambdas.data(), condensed_sizes.data());

    ML::HDBSCAN::detail::Extract::extract_clusters(
      handle, condensed_tree, params.n_row, labels.data(), stabilities.data(),
      probabilities.data(), params.cluster_selection_method,
      params.allow_single_cluster, 0, params.cluster_selection_epsilon);

    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    raft::print_device_vector("probabilities", probabilities.data(),
                              params.n_row, std::cout);
    raft::print_device_vector("labels", labels.data(), params.n_row, std::cout);

    ASSERT_TRUE(raft::devArrMatch(
      probabilities.data(), params.probabilities.data(), params.n_row,
      raft::CompareApprox<float>(1e-4), handle.get_stream()));

    rmm::device_uvector<IdxT> labels_ref(params.n_row, handle.get_stream());
    raft::update_device(labels_ref.data(), params.labels.data(), params.n_row,
                        handle.get_stream());
    score = MLCommon::Metrics::compute_adjusted_rand_index(
      labels.data(), labels_ref.data(), params.n_row,
      handle.get_device_allocator(), handle.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  ClusterSelectionInputs<T, IdxT> params;
  T score;
};

typedef ClusterSelectionTest<float, int> ClusterSelectionTestF_Int;
TEST_P(ClusterSelectionTestF_Int, Result) { EXPECT_TRUE(score == 1.0); }

INSTANTIATE_TEST_CASE_P(ClusterSelectionTest, ClusterSelectionTestF_Int,
                        ::testing::ValuesIn(cluster_selection_inputs));

}  // namespace HDBSCAN
}  // end namespace ML
