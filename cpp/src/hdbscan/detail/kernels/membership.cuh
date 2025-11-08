/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Membership {

template <typename value_idx, typename value_t>
struct probabilities_functor {
 public:
  probabilities_functor(value_t* probabilities_,
                        const value_t* deaths_,
                        const value_idx* children_,
                        const value_t* lambdas_,
                        const value_idx* labels_,
                        const value_idx root_cluster_)
    : probabilities(probabilities_),
      deaths(deaths_),
      children(children_),
      lambdas(lambdas_),
      labels(labels_),
      root_cluster(root_cluster_)
  {
  }

  __device__ void operator()(const value_idx& idx)
  {
    auto child = children[idx];

    // intermediate nodes
    if (child >= root_cluster) { return; }

    auto cluster = labels[child];

    // noise
    if (cluster == -1) { return; }

    auto cluster_death = deaths[cluster];
    auto child_lambda  = lambdas[idx];
    if (cluster_death == 0.0 || isnan(child_lambda)) {
      probabilities[child] = 1.0;
    } else {
      auto min_lambda      = min(child_lambda, cluster_death);
      probabilities[child] = min_lambda / cluster_death;
    }
  }

 private:
  value_t* probabilities;
  const value_t *deaths, *lambdas;
  const value_idx *children, *labels, root_cluster;
};

};  // namespace Membership
};  // namespace detail
};  // namespace HDBSCAN
};  // namespace ML
