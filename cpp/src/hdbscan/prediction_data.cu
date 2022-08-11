#include "detail/utils.h"

#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>

#include <cuml/cluster/hdbscan.hpp>

#include <raft/sparse/convert/csr.hpp>
#include <raft/sparse/op/sort.hpp>
#include <raft/matrix/math.hpp>

#include <algorithm>
#include <cmath>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
namespace ML {
namespace HDBSCAN {
namespace Common {

template <typename value_idx, typename value_t>
void PredictionData<value_idx, value_t>::allocate(const raft::handle_t& handle,
                                                                   value_idx n_exemplars_,
                                                                   value_idx n_clusters_,
                                                                   value_idx n_selected_clusters_)
{
  this->n_exemplars         = n_exemplars_;
  this->n_selected_clusters = n_selected_clusters_;
  exemplar_idx.resize(n_exemplars, handle.get_stream());
  exemplar_label_offsets.resize(n_selected_clusters + 1, handle.get_stream());
  deaths.resize(n_clusters_, handle.get_stream());
  selected_clusters.resize(n_selected_clusters, handle.get_stream());
}

void build_prediction_data(const raft::handle_t& handle,
                           CondensedHierarchy<int, float>& condensed_tree,
                           int* labels,
                           int* label_map,
                           int n_selected_clusters,
                           PredictionData<int, float>& prediction_data)
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  auto counting = thrust::make_counting_iterator<int>(0);

  auto parents    = condensed_tree.get_parents();
  auto children   = condensed_tree.get_children();
  auto lambdas    = condensed_tree.get_lambdas();
  auto n_edges    = condensed_tree.get_n_edges();
  auto n_clusters = condensed_tree.get_n_clusters();
  auto n_leaves   = condensed_tree.get_n_leaves();
  auto sizes      = condensed_tree.get_sizes();

  // first compute the death of each cluster in the condensed hierarchy
  rmm::device_uvector<float> deaths(n_clusters, stream);

  rmm::device_uvector<int> sorted_parents(n_edges, stream);
  raft::copy_async(sorted_parents.data(), parents, n_edges, stream);

  rmm::device_uvector<int> sorted_parents_offsets(n_clusters + 1, stream);
  detail::Utils::parent_csr(handle, condensed_tree, sorted_parents.data(), sorted_parents_offsets.data());

  // this is to find maximum lambdas of all children under a parent
  detail::Utils::cub_segmented_reduce(
    lambdas,
    deaths.data(),
    n_clusters,
    sorted_parents_offsets.data(),
    stream,
    cub::DeviceSegmentedReduce::Max<const float*, float*, const int*, const int*>);

  rmm::device_uvector<int> is_leaf_cluster(n_clusters, stream);
  thrust::fill(exec_policy, is_leaf_cluster.begin(), is_leaf_cluster.end(), 1);

  auto leaf_cluster_op =
    [is_leaf_cluster = is_leaf_cluster.data(), parents, sizes, n_leaves] __device__(auto idx) {
      if (sizes[idx] > 1) { is_leaf_cluster[parents[idx] - n_leaves] = 0; }
      return;
    };

  thrust::for_each(exec_policy, counting, counting + n_edges, leaf_cluster_op);

  rmm::device_uvector<int> is_exemplar(n_leaves, stream);
  rmm::device_uvector<int> exemplar_idx(n_leaves, stream);
  rmm::device_uvector<int> exemplar_label_offsets(n_selected_clusters + 1, stream);
  rmm::device_uvector<int> selected_clusters(n_selected_clusters, stream);

  // classify whether or not a point is an exemplar point using the death values
  auto exemplar_op = [is_exemplar = is_exemplar.data(),
                      lambdas,
                      is_leaf_cluster = is_leaf_cluster.data(),
                      parents,
                      children,
                      n_leaves,
                      deaths = deaths.data()] __device__(auto idx) {
    if (children[idx] < n_leaves) {
      is_exemplar[children[idx]] = (is_leaf_cluster[parents[idx] - n_leaves] &&
                                    lambdas[idx] == deaths[parents[idx] - n_leaves]);
      return;
    }
  };

  thrust::for_each(exec_policy, counting, counting + n_edges, exemplar_op);

  int n_exemplars = thrust::count_if(exec_policy,
    is_exemplar.begin(),
    is_exemplar.end(),
    [] __device__(auto idx) { return idx; }); 		

  prediction_data.allocate(handle,
    n_exemplars,
n_clusters,
n_selected_clusters);

  auto exemplar_idx_end_ptr = thrust::copy_if(
    exec_policy,
    counting,
    counting + n_leaves,
    prediction_data.get_exemplar_idx(),
    [is_exemplar = is_exemplar.data()] __device__(auto idx) { return is_exemplar[idx]; });

  // use the exemplar labels to fetch the set of selected clusters from the condensed hierarchy
  rmm::device_uvector<int> exemplar_labels(n_exemplars, stream);

  thrust::transform(exec_policy,
    prediction_data.get_exemplar_idx(),
    prediction_data.get_exemplar_idx() + n_exemplars,
                    exemplar_labels.data(),
                    [labels] __device__(auto idx) { return labels[idx]; });

  thrust::sort_by_key(
    exec_policy, exemplar_labels.data(), exemplar_labels.data() + n_exemplars, prediction_data.get_exemplar_idx());

  rmm::device_uvector<int> converted_exemplar_labels(n_exemplars, stream);
  thrust::transform(exec_policy,
                    exemplar_labels.begin(),
                    exemplar_labels.end(),
                    converted_exemplar_labels.data(),
                    [label_map] __device__(auto idx) { return label_map[idx]; });

  raft::sparse::convert::sorted_coo_to_csr(converted_exemplar_labels.data(),
                                           n_exemplars,
                                           prediction_data.get_exemplar_label_offsets(),
                                           n_selected_clusters + 1,
                                           stream);

  thrust::transform(exec_policy,
    prediction_data.get_exemplar_label_offsets(),
    prediction_data.get_exemplar_label_offsets() + n_selected_clusters,
                    prediction_data.get_selected_clusters(),
                    [exemplar_labels = exemplar_labels.data(), n_leaves] __device__(auto idx) {
                      return exemplar_labels[idx] + n_leaves;
                    });
}

}; // end namespace Common
}; // end namespace HDBSCAN
};  // end namespace ML