
#include "cuML.hpp"

#pragma once

namespace ML {

void TSNE_fit(
  const cumlHandle &handle, const float *X, float *Y, const int n, const int p,
  const int n_components = 2, int n_neighbors = 30,

  float perplexity = 30.0f, const int perplexity_max_iter = 100,
  const int perplexity_tol = 1e-5,

  const float early_exaggeration = 12.0f, const int exaggeration_iter = 150,

  const float min_gain = 0.01f, const float gains_add = 0.3f,
  const float gains_mult = 0.7f,
  // Original TSNE paper has gains_add = 0.2 and gains_mult = 0.8

  const float eta = 500.0f, const int max_iter = 500,
  const float pre_momentum = 0.99, const float post_momentum = 0.5,
  // Original TSNE paper has pre_momentum = 0.8. We also add Momentum decay

  const long long seed = -1, const bool initialize_embeddings = true,
  const bool verbose = true);

}
