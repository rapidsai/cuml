/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/manifold/spectral_embedding.hpp>

#include <cuvs/preprocessing/spectral_embedding.hpp>

namespace ML::SpectralEmbedding {

inline cuvs::preprocessing::spectral_embedding::params to_cuvs(params const& config)
{
  cuvs::preprocessing::spectral_embedding::params params;

  params.n_components   = config.n_components;
  params.n_neighbors    = config.n_neighbors;
  params.norm_laplacian = config.norm_laplacian;
  params.drop_first     = config.drop_first;
  params.seed           = config.seed;

  return params;
}

}  // namespace ML::SpectralEmbedding
