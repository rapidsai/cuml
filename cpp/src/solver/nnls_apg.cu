/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nnls_apg.cuh"

#include <cuml/solvers/nnls.hpp>

#include <raft/core/handle.hpp>

namespace ML {
namespace Solver {

int nnlsApg(raft::handle_t&     handle,
            const float*        A,
            int                 n_rows,
            int                 n_cols,
            const float*        b,
            float*              x,
            const NnlsApgParams& params)
{
  return detail::nnls_apg_impl<float>(handle, A, n_rows, n_cols, b, x, params);
}

int nnlsApg(raft::handle_t&     handle,
            const double*       A,
            int                 n_rows,
            int                 n_cols,
            const double*       b,
            double*             x,
            const NnlsApgParams& params)
{
  return detail::nnls_apg_impl<double>(handle, A, n_rows, n_cols, b, x, params);
}

}  // namespace Solver
}  // namespace ML
