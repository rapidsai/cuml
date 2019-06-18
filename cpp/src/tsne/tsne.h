
#include "cuML.hpp"

#pragma once

namespace ML {

void TSNE(const cumlHandle &handle, const float *X, float *Y, const int n,
          const int p, const int n_components, int n_neighbors,
          const float perplexity, const int perplexity_epochs,
          const int perplexity_tol, const float early_exaggeration,
          const int exaggeration_iter, const float min_gain, const float eta,
          const int epochs, const float pre_momentum, const float post_momentum,
          const long long seed, const bool initialize_embeddings);

}
