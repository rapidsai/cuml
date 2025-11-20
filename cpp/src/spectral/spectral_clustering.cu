/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/cluster/spectral_clustering.hpp>

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/cluster/spectral.hpp>

namespace cuvs::cluster::spectral {
struct params;
}  // end namespace cuvs::cluster::spectral

namespace ML {
namespace SpectralClustering {

cuvs::cluster::spectral::params to_cuvs(ML::SpectralClustering::params& config)
{
  cuvs::cluster::spectral::params cuvs_params;
  cuvs_params.n_clusters   = config.n_clusters;
  cuvs_params.n_components = config.n_components;
  cuvs_params.n_init       = config.n_init;
  cuvs_params.n_neighbors  = config.n_neighbors;
  cuvs_params.tolerance    = config.eigen_tol;
  cuvs_params.rng_state    = raft::random::RngState(config.seed);

  return cuvs_params;
}

void fit_predict(raft::resources const& handle,
                 params config,
                 raft::device_coo_matrix_view<float, int, int, int> connectivity_graph,
                 raft::device_vector_view<int, int> labels)
{
  cuvs::cluster::spectral::fit_predict(handle, to_cuvs(config), connectivity_graph, labels);
}

void fit_predict(raft::resources const& handle,
                 params config,
                 raft::device_vector_view<int, int> rows,
                 raft::device_vector_view<int, int> cols,
                 raft::device_vector_view<float, int> vals,
                 raft::device_vector_view<int, int> labels)
{
  auto connectivity_graph_view = raft::make_device_coo_matrix_view<float, int, int, int>(
    vals.data_handle(),
    raft::make_device_coordinate_structure_view<int, int, int>(
      rows.data_handle(), cols.data_handle(), labels.size(), labels.size(), vals.size()));

  ML::SpectralClustering::fit_predict(handle, config, connectivity_graph_view, labels);
}

}  // namespace SpectralClustering
}  // namespace ML
