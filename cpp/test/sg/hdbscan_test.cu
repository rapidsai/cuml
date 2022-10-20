/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <vector>

#include <cuml/cluster/hdbscan.hpp>
#include <hdbscan/detail/condense.cuh>
#include <hdbscan/detail/extract.cuh>
#include <hdbscan/detail/predict.cuh>
#include <hdbscan/detail/reachability.cuh>
#include <hdbscan/detail/soft_clustering.cuh>
#include <hdbscan/detail/utils.h>

#include <raft/spatial/knn/specializations.hpp>
#include <raft/stats/adjusted_rand_index.hpp>

#include <raft/cluster/detail/agglomerative.cuh>

#include <raft/distance/distance_types.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/op/sort.cuh>
#include <rmm/device_uvector.hpp>

#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include "../prims/test_utils.h"

namespace ML {
namespace HDBSCAN {

using namespace std;

template <typename T, typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const HDBSCANInputs<T, IdxT>& dims)
{
  return os;
}

template <typename T, typename IdxT>
class HDBSCANTest : public ::testing::TestWithParam<HDBSCANInputs<T, IdxT>> {
 protected:
  void basicTest()
  {
    raft::handle_t handle;

    params = ::testing::TestWithParam<HDBSCANInputs<T, IdxT>>::GetParam();

    rmm::device_uvector<T> data(params.n_row * params.n_col, handle.get_stream());

    // Allocate result labels and expected labels on device
    rmm::device_uvector<IdxT> labels_ref(params.n_row, handle.get_stream());

    raft::copy(data.data(), params.data.data(), data.size(), handle.get_stream());
    raft::copy(labels_ref.data(), params.expected_labels.data(), params.n_row, handle.get_stream());

    rmm::device_uvector<IdxT> out_children(params.n_row * 2, handle.get_stream());
    rmm::device_uvector<T> out_deltas(params.n_row, handle.get_stream());

    rmm::device_uvector<IdxT> out_sizes(params.n_row * 2, handle.get_stream());

    rmm::device_uvector<IdxT> out_labels(params.n_row, handle.get_stream());

    rmm::device_uvector<IdxT> mst_src(params.n_row - 1, handle.get_stream());
    rmm::device_uvector<IdxT> mst_dst(params.n_row - 1, handle.get_stream());
    rmm::device_uvector<T> mst_weights(params.n_row - 1, handle.get_stream());

    rmm::device_uvector<T> out_probabilities(params.n_row, handle.get_stream());

    Logger::get().setLevel(CUML_LEVEL_DEBUG);

    HDBSCAN::Common::hdbscan_output<IdxT, T> out(handle,
                                                 params.n_row,
                                                 out_labels.data(),
                                                 out_probabilities.data(),
                                                 out_children.data(),
                                                 out_sizes.data(),
                                                 out_deltas.data(),
                                                 mst_src.data(),
                                                 mst_dst.data(),
                                                 mst_weights.data());

    HDBSCAN::Common::HDBSCANParams hdbscan_params;
    hdbscan_params.min_cluster_size = params.min_cluster_size;
    hdbscan_params.min_samples      = params.min_pts;

    HDBSCAN::Common::PredictionData<IdxT, T> prediction_data_(handle, params.n_row, params.n_col);

    hdbscan(handle,
            data.data(),
            params.n_row,
            params.n_col,
            raft::distance::DistanceType::L2SqrtExpanded,
            hdbscan_params,
            out,
            prediction_data_);

    handle.sync_stream(handle.get_stream());

    score = raft::stats::adjusted_rand_index(
      out.get_labels(), labels_ref.data(), params.n_row, handle.get_stream());

    if (score < 0.85) {
      std::cout << "Test failed. score=" << score << std::endl;
      raft::print_device_vector("actual labels", out.get_labels(), params.n_row, std::cout);
      raft::print_device_vector("expected labels", labels_ref.data(), params.n_row, std::cout);
    }
  }

  void SetUp() override { basicTest(); }

 protected:
  HDBSCANInputs<T, IdxT> params;
  IdxT* labels_ref;

  double score;
};

typedef HDBSCANTest<float, int> HDBSCANTestF_Int;
TEST_P(HDBSCANTestF_Int, Result) { EXPECT_TRUE(score >= 0.85); }

INSTANTIATE_TEST_CASE_P(HDBSCANTest, HDBSCANTestF_Int, ::testing::ValuesIn(hdbscan_inputsf2));

template <typename T, typename IdxT>
class ClusterCondensingTest : public ::testing::TestWithParam<ClusterCondensingInputs<T, IdxT>> {
 protected:
  void basicTest()
  {
    raft::handle_t handle;

    params = ::testing::TestWithParam<ClusterCondensingInputs<T, IdxT>>::GetParam();

    rmm::device_uvector<IdxT> mst_src(params.n_row - 1, handle.get_stream());
    rmm::device_uvector<IdxT> mst_dst(params.n_row - 1, handle.get_stream());
    rmm::device_uvector<T> mst_data(params.n_row - 1, handle.get_stream());

    raft::copy(mst_src.data(), params.mst_src.data(), params.mst_src.size(), handle.get_stream());

    raft::copy(mst_dst.data(), params.mst_dst.data(), params.mst_dst.size(), handle.get_stream());

    raft::copy(
      mst_data.data(), params.mst_data.data(), params.mst_data.size(), handle.get_stream());

    rmm::device_uvector<IdxT> expected_device(params.expected.size(), handle.get_stream());
    raft::copy(
      expected_device.data(), params.expected.data(), params.expected.size(), handle.get_stream());

    rmm::device_uvector<IdxT> out_children(params.n_row * 2, handle.get_stream());

    rmm::device_uvector<IdxT> out_size(params.n_row, handle.get_stream());

    rmm::device_uvector<T> out_delta(params.n_row, handle.get_stream());

    Logger::get().setLevel(CUML_LEVEL_DEBUG);

    raft::sparse::op::coo_sort_by_weight(
      mst_src.data(), mst_dst.data(), mst_data.data(), (IdxT)mst_src.size(), handle.get_stream());

    /**
     * Build dendrogram of MST
     */
    raft::cluster::detail::build_dendrogram_host(handle,
                                                 mst_src.data(),
                                                 mst_dst.data(),
                                                 mst_data.data(),
                                                 params.n_row - 1,
                                                 out_children.data(),
                                                 out_delta.data(),
                                                 out_size.data());

    /**
     * Condense Hierarchy
     */
    HDBSCAN::Common::CondensedHierarchy<IdxT, T> condensed_tree(handle, params.n_row);
    HDBSCAN::detail::Condense::build_condensed_hierarchy(handle,
                                                         out_children.data(),
                                                         out_delta.data(),
                                                         out_size.data(),
                                                         params.min_cluster_size,
                                                         params.n_row,
                                                         condensed_tree);

    handle.sync_stream(handle.get_stream());

    rmm::device_uvector<IdxT> labels(params.n_row, handle.get_stream());
    rmm::device_uvector<T> stabilities(condensed_tree.get_n_clusters(), handle.get_stream());
    rmm::device_uvector<T> probabilities(params.n_row, handle.get_stream());
    rmm::device_uvector<IdxT> label_map(params.n_row, handle.get_stream());

    HDBSCAN::detail::Extract::extract_clusters(handle,
                                               condensed_tree,
                                               params.n_row,
                                               labels.data(),
                                               stabilities.data(),
                                               probabilities.data(),
                                               label_map.data(),
                                               HDBSCAN::Common::CLUSTER_SELECTION_METHOD::EOM,
                                               false);

    //    CUML_LOG_DEBUG("Evaluating results");
    //    if (params.expected.size() == params.n_row) {
    //      score = MLCommon::Metrics::compute_adjusted_rand_index(
    //        labels.data(), expected_device.data(), params.n_row,
    //        handle.get_stream());
    //    } else {
    //      score = 1.0;
    //    }
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  ClusterCondensingInputs<T, IdxT> params;

  double score;
};

typedef ClusterCondensingTest<float, int> ClusterCondensingTestF_Int;
TEST_P(ClusterCondensingTestF_Int, Result) { EXPECT_TRUE(score == 1.0); }

// This will be reactivate in 21.08 with better, contrived examples to
// test Cluster Condensation correctly
// INSTANTIATE_TEST_CASE_P(ClusterCondensingTest, ClusterCondensingTestF_Int,
//                         ::testing::ValuesIn(cluster_condensing_inputs));

template <typename T, typename IdxT>
class ClusterSelectionTest : public ::testing::TestWithParam<ClusterSelectionInputs<T, IdxT>> {
 protected:
  void basicTest()
  {
    raft::handle_t handle;

    params = ::testing::TestWithParam<ClusterSelectionInputs<T, IdxT>>::GetParam();

    Logger::get().setLevel(CUML_LEVEL_DEBUG);

    rmm::device_uvector<IdxT> condensed_parents(params.condensed_parents.size(),
                                                handle.get_stream());
    rmm::device_uvector<IdxT> condensed_children(params.condensed_children.size(),
                                                 handle.get_stream());
    rmm::device_uvector<T> condensed_lambdas(params.condensed_lambdas.size(), handle.get_stream());
    rmm::device_uvector<IdxT> condensed_sizes(params.condensed_sizes.size(), handle.get_stream());

    // outputs
    rmm::device_uvector<T> stabilities(params.n_row, handle.get_stream());
    rmm::device_uvector<T> probabilities(params.n_row, handle.get_stream());
    rmm::device_uvector<IdxT> labels(params.n_row, handle.get_stream());

    raft::copy(condensed_parents.data(),
               params.condensed_parents.data(),
               condensed_parents.size(),
               handle.get_stream());

    raft::copy(condensed_children.data(),
               params.condensed_children.data(),
               condensed_children.size(),
               handle.get_stream());

    raft::copy(condensed_lambdas.data(),
               params.condensed_lambdas.data(),
               condensed_lambdas.size(),
               handle.get_stream());

    raft::copy(condensed_sizes.data(),
               params.condensed_sizes.data(),
               condensed_sizes.size(),
               handle.get_stream());

    ML::HDBSCAN::Common::CondensedHierarchy<IdxT, T> condensed_tree(handle,
                                                                    params.n_row,
                                                                    params.condensed_parents.size(),
                                                                    condensed_parents.data(),
                                                                    condensed_children.data(),
                                                                    condensed_lambdas.data(),
                                                                    condensed_sizes.data());

    rmm::device_uvector<IdxT> label_map(params.n_row, handle.get_stream());

    ML::HDBSCAN::detail::Extract::extract_clusters(handle,
                                                   condensed_tree,
                                                   params.n_row,
                                                   labels.data(),
                                                   stabilities.data(),
                                                   probabilities.data(),
                                                   label_map.data(),
                                                   params.cluster_selection_method,
                                                   params.allow_single_cluster,
                                                   0,
                                                   params.cluster_selection_epsilon);

    handle.sync_stream(handle.get_stream());

    ASSERT_TRUE(raft::devArrMatch(probabilities.data(),
                                  params.probabilities.data(),
                                  params.n_row,
                                  raft::CompareApprox<float>(1e-4),
                                  handle.get_stream()));

    rmm::device_uvector<IdxT> labels_ref(params.n_row, handle.get_stream());
    raft::update_device(labels_ref.data(), params.labels.data(), params.n_row, handle.get_stream());
    score = raft::stats::adjusted_rand_index(
      labels.data(), labels_ref.data(), params.n_row, handle.get_stream());
    handle.sync_stream(handle.get_stream());
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  ClusterSelectionInputs<T, IdxT> params;
  T score;
};

typedef ClusterSelectionTest<float, int> ClusterSelectionTestF_Int;
TEST_P(ClusterSelectionTestF_Int, Result) { EXPECT_TRUE(score == 1.0); }

INSTANTIATE_TEST_CASE_P(ClusterSelectionTest,
                        ClusterSelectionTestF_Int,
                        ::testing::ValuesIn(cluster_selection_inputs));

// This test was constructed in the following manner: The same condensed tree and set of selected
// clusters need to be passed to the reference implementation and then compare the results from
// cuML and the reference implementation for an approximate match of probabilities. To fetch the
// condensed hierarchy in the same format as required by the reference implementation, a simple
// python script can be written:
// 1. Print the parents, children, lambdas and sizes array of the condensed hierarchy.
// 2. Convert them into a list ``condensed_tree`` of tuples where each tuples is of the form.
//    ``(parents[i], children[i], lambdas[i], sizes[i])``
// 3. Convert the list into a numpy array with the following command:
//    ``condensed_tree_array = np.array(condened_tree, dtype=[('parent', np.intp), ('child',
//                                      np.intp), ('lambda_val', float), ('child_size',
//                                      np.intp)])``
// 4. Store it in a pickle file.
// The reference source code is modified in the following way: Edit the raw tree in the init
// function of the PredictionData object in prediction.py by loading it from the pickle file. Also
// edit the selected clusters array. Do the same in the all_points_membership_vectors function and
// the approximate_predict functions.
template <typename T, typename IdxT>
class SoftClusteringTest : public ::testing::TestWithParam<SoftClusteringInputs<T, IdxT>> {
 protected:
  void basicTest()
  {
    raft::handle_t handle;

    params = ::testing::TestWithParam<SoftClusteringInputs<T, IdxT>>::GetParam();

    rmm::device_uvector<IdxT> condensed_parents(params.condensed_parents.size(),
                                                handle.get_stream());
    rmm::device_uvector<IdxT> condensed_children(params.condensed_children.size(),
                                                 handle.get_stream());
    rmm::device_uvector<T> condensed_lambdas(params.condensed_lambdas.size(), handle.get_stream());
    rmm::device_uvector<IdxT> condensed_sizes(params.condensed_sizes.size(), handle.get_stream());

    raft::copy(condensed_parents.data(),
               params.condensed_parents.data(),
               condensed_parents.size(),
               handle.get_stream());

    raft::copy(condensed_children.data(),
               params.condensed_children.data(),
               condensed_children.size(),
               handle.get_stream());

    raft::copy(condensed_lambdas.data(),
               params.condensed_lambdas.data(),
               condensed_lambdas.size(),
               handle.get_stream());

    raft::copy(condensed_sizes.data(),
               params.condensed_sizes.data(),
               condensed_sizes.size(),
               handle.get_stream());

    rmm::device_uvector<T> data(params.n_row * params.n_col, handle.get_stream());
    raft::copy(data.data(), params.data.data(), data.size(), handle.get_stream());

    ML::HDBSCAN::Common::CondensedHierarchy<IdxT, T> condensed_tree(handle,
                                                                    params.n_row,
                                                                    params.condensed_parents.size(),
                                                                    condensed_parents.data(),
                                                                    condensed_children.data(),
                                                                    condensed_lambdas.data(),
                                                                    condensed_sizes.data());

    rmm::device_uvector<IdxT> label_map(params.n_row, handle.get_stream());

    // intermediate outputs
    rmm::device_uvector<T> stabilities(params.n_row, handle.get_stream());
    rmm::device_uvector<T> probabilities(params.n_row, handle.get_stream());
    rmm::device_uvector<IdxT> labels(params.n_row, handle.get_stream());

    int n_selected_clusters =
      ML::HDBSCAN::detail::Extract::extract_clusters(handle,
                                                     condensed_tree,
                                                     params.n_row,
                                                     labels.data(),
                                                     stabilities.data(),
                                                     probabilities.data(),
                                                     label_map.data(),
                                                     params.cluster_selection_method,
                                                     params.allow_single_cluster,
                                                     0,
                                                     params.cluster_selection_epsilon);

    rmm::device_uvector<T> membership_vec(params.n_row * n_selected_clusters, handle.get_stream());

    ML::HDBSCAN::Common::PredictionData<IdxT, T> prediction_data_(
      handle, params.n_row, params.n_col);

    ML::HDBSCAN::Common::build_prediction_data(handle,
                                               condensed_tree,
                                               labels.data(),
                                               label_map.data(),
                                               n_selected_clusters,
                                               prediction_data_);

    ML::compute_all_points_membership_vectors(handle,
                                              condensed_tree,
                                              prediction_data_,
                                              data.data(),
                                              raft::distance::DistanceType::L2SqrtExpanded,
                                              membership_vec.data());

    ASSERT_TRUE(raft::devArrMatch(membership_vec.data(),
                                  params.expected_probabilities.data(),
                                  params.n_row * n_selected_clusters,
                                  raft::CompareApprox<float>(1e-5),
                                  handle.get_stream()));
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  SoftClusteringInputs<T, IdxT> params;
  // T score;
};

typedef SoftClusteringTest<float, int> SoftClusteringTestF_Int;
TEST_P(SoftClusteringTestF_Int, Result) { EXPECT_TRUE(true); }

INSTANTIATE_TEST_CASE_P(SoftClusteringTest,
                        SoftClusteringTestF_Int,
                        ::testing::ValuesIn(soft_clustering_inputs));

template <typename T, typename IdxT>
class ApproximatePredictTest : public ::testing::TestWithParam<ApproximatePredictInputs<T, IdxT>> {
 public:
  void transformLabels(const raft::handle_t& handle, IdxT* labels, IdxT* label_map, IdxT m)
  {
    thrust::transform(
      handle.get_thrust_policy(), labels, labels + m, labels, [label_map] __device__(IdxT label) {
        if (label != -1) return label_map[label];
        return -1;
      });
  }

 protected:
  void basicTest()
  {
    raft::handle_t handle;

    params = ::testing::TestWithParam<ApproximatePredictInputs<T, IdxT>>::GetParam();

    rmm::device_uvector<IdxT> condensed_parents(params.condensed_parents.size(),
                                                handle.get_stream());
    rmm::device_uvector<IdxT> condensed_children(params.condensed_children.size(),
                                                 handle.get_stream());
    rmm::device_uvector<T> condensed_lambdas(params.condensed_lambdas.size(), handle.get_stream());
    rmm::device_uvector<IdxT> condensed_sizes(params.condensed_sizes.size(), handle.get_stream());

    raft::copy(condensed_parents.data(),
               params.condensed_parents.data(),
               condensed_parents.size(),
               handle.get_stream());

    raft::copy(condensed_children.data(),
               params.condensed_children.data(),
               condensed_children.size(),
               handle.get_stream());

    raft::copy(condensed_lambdas.data(),
               params.condensed_lambdas.data(),
               condensed_lambdas.size(),
               handle.get_stream());

    raft::copy(condensed_sizes.data(),
               params.condensed_sizes.data(),
               condensed_sizes.size(),
               handle.get_stream());

    rmm::device_uvector<T> data(params.n_row * params.n_col, handle.get_stream());
    raft::copy(data.data(), params.data.data(), data.size(), handle.get_stream());

    rmm::device_uvector<T> points_to_predict(params.n_points_to_predict * params.n_col,
                                             handle.get_stream());
    raft::copy(points_to_predict.data(),
               params.points_to_predict.data(),
               points_to_predict.size(),
               handle.get_stream());

    ML::HDBSCAN::Common::CondensedHierarchy<IdxT, T> condensed_tree(handle,
                                                                    params.n_row,
                                                                    params.condensed_parents.size(),
                                                                    condensed_parents.data(),
                                                                    condensed_children.data(),
                                                                    condensed_lambdas.data(),
                                                                    condensed_sizes.data());

    rmm::device_uvector<IdxT> label_map(params.n_row, handle.get_stream());

    // intermediate outputs
    rmm::device_uvector<T> stabilities(params.n_row, handle.get_stream());
    rmm::device_uvector<T> probabilities(params.n_row, handle.get_stream());
    rmm::device_uvector<IdxT> labels(params.n_row, handle.get_stream());

    int n_selected_clusters =
      ML::HDBSCAN::detail::Extract::extract_clusters(handle,
                                                     condensed_tree,
                                                     params.n_row,
                                                     labels.data(),
                                                     stabilities.data(),
                                                     probabilities.data(),
                                                     label_map.data(),
                                                     params.cluster_selection_method,
                                                     params.allow_single_cluster,
                                                     0,
                                                     params.cluster_selection_epsilon);

    ML::HDBSCAN::Common::PredictionData<IdxT, T> pred_data(handle, params.n_row, params.n_col);

    auto stream = handle.get_stream();
    rmm::device_uvector<IdxT> mutual_reachability_indptr(params.n_row + 1, stream);
    raft::sparse::COO<T, IdxT> mutual_reachability_coo(stream,
                                                       (params.min_samples + 1) * params.n_row * 2);

    ML::HDBSCAN::detail::Reachability::mutual_reachability_graph(
      handle,
      data.data(),
      (size_t)params.n_row,
      (size_t)params.n_col,
      raft::distance::DistanceType::L2SqrtExpanded,
      params.min_samples + 1,
      (float)1.0,
      mutual_reachability_indptr.data(),
      pred_data.get_core_dists(),
      mutual_reachability_coo);

    ML::HDBSCAN::Common::build_prediction_data(
      handle, condensed_tree, labels.data(), label_map.data(), n_selected_clusters, pred_data);

    // outputs
    rmm::device_uvector<IdxT> out_labels(params.n_points_to_predict, handle.get_stream());
    rmm::device_uvector<T> out_probabilities(params.n_points_to_predict, handle.get_stream());

    transformLabels(handle, labels.data(), label_map.data(), params.n_row);

    ML::out_of_sample_predict(handle,
                              condensed_tree,
                              pred_data,
                              const_cast<float*>(data.data()),
                              labels.data(),
                              const_cast<float*>(points_to_predict.data()),
                              (size_t)params.n_points_to_predict,
                              raft::distance::DistanceType::L2SqrtExpanded,
                              params.min_samples,
                              out_labels.data(),
                              out_probabilities.data());

    handle.sync_stream(handle.get_stream());
    cudaDeviceSynchronize();

    ASSERT_TRUE(raft::devArrMatch(out_labels.data(),
                                  params.expected_labels.data(),
                                  params.n_points_to_predict,
                                  raft::Compare<int>(),
                                  handle.get_stream()));

    ASSERT_TRUE(raft::devArrMatch(out_probabilities.data(),
                                  params.expected_probabilities.data(),
                                  params.n_points_to_predict,
                                  raft::CompareApprox<float>(1e-2),
                                  handle.get_stream()));
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  ApproximatePredictInputs<T, IdxT> params;
  // T score;
};

typedef ApproximatePredictTest<float, int> ApproximatePredictTestF_Int;
TEST_P(ApproximatePredictTestF_Int, Result) { EXPECT_TRUE(true); }

INSTANTIATE_TEST_CASE_P(ApproximatePredictTest,
                        ApproximatePredictTestF_Int,
                        ::testing::ValuesIn(approximate_predict_inputs));
}  // namespace HDBSCAN
}  // end namespace ML
