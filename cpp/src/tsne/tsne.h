
#include "cuML.hpp"

#pragma once

namespace ML {

void TSNE_fit(
  const cumlHandle &handle, const float *X, float *Y, const int n, const int p,
  const int n_components = 2, int n_neighbors = 30,

  float perplexity = 30.0f, const int perplexity_max_iter = 100,
  const int perplexity_tol = 1e-5,

  const float early_exaggeration = 12.0f, const int exaggeration_iter = 250,

  const float min_gain = 0.01f, const double min_grad_norm = 1e-4,
  const float eta = 500.0f, const int max_iter = 1000,
  const float pre_momentum = 0.99, const float post_momentum = 0.5,
  // Original TSNE pre = 0.8 and most = 0.5. We also add momentum decay of 0.001

  const long long seed = -1, const bool initialize_embeddings = true,
  const bool verbose = true);

}
