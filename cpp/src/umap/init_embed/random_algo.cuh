/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/manifold/umapparams.h>

#include <raft/random/rng.cuh>

#include <stdint.h>

namespace UMAPAlgo {
namespace InitEmbed {
namespace RandomInit {

using namespace ML;

template <typename T, typename nnz_t>
void launcher(nnz_t n, int d, UMAPParams* params, T* embedding, cudaStream_t stream)
{
  uint64_t seed = params->random_state;

  raft::random::Rng r(seed);
  r.uniform<T>(embedding, n * params->n_components, -10, 10, stream);
}
}  // namespace RandomInit
}  // namespace InitEmbed
};  // namespace UMAPAlgo
