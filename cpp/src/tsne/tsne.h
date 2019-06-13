
#include "cuML.hpp"

#pragma once

namespace ML {

void TSNE(const cumlHandle &handle, const float *X, float *Y, const int n,
          const int p, const int n_components = 2, const int n_neighbors = 90,
          const float perplexity = 30.0f, const int perplexity_epochs = 100,
          const int perplexity_tol = 1e-5,
          const float early_exaggeration = 12.0f,
          const int exaggeration_iter = 250, const float min_gain = 0.01f,
          const float eta = 500.0f, const int epochs = 150,
          const float pre_momentum = 0.8, const float post_momentum = 0.5,
          const long long seed = -1);
}
