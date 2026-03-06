/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "algo.cuh"

#include <cuml/manifold/umapparams.h>

#include <raft/sparse/coo.hpp>

namespace UMAPAlgo {

namespace SimplSetEmbed {

using namespace ML;

template <typename T, typename nnz_t, int TPB_X>
void run(int m,
         int n,
         raft::sparse::COO<T>* coo,
         UMAPParams* params,
         T* embedding,
         int n_epochs,
         cudaStream_t stream,
         int algorithm               = 0,
         DensMap::DensMapData<T>* dm = nullptr)
{
  switch (algorithm) {
    case 0:
      SimplSetEmbed::Algo::launcher<T, nnz_t, TPB_X>(
        m, n, coo, params, embedding, n_epochs, stream, dm);
  }
}
}  // namespace SimplSetEmbed
}  // namespace UMAPAlgo
