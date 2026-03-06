/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/linalg/pca_types.hpp>

#include <cstdint>

namespace ML {

using solver    = raft::linalg::solver;
using mg_solver = raft::linalg::solver;

class params {
 public:
  std::size_t n_rows;
  std::size_t n_cols;
  int gpu_id = 0;
};

class paramsSolver : public params {
 public:
  // math_t tol = 0.0;
  float tol                  = 0.0;
  std::uint32_t n_iterations = 15;
  int verbose                = 0;
};

template <typename enum_solver = solver>
class paramsTSVDTemplate : public paramsSolver {
 public:
  std::size_t n_components = 1;
  enum_solver algorithm    = enum_solver::COV_EIG_DQ;
};

/**
 * @brief structure for pca parameters. Ref:
 * http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
 * @param n_components: Number of components to keep. if n_components is not set all components are
 * kept:
 * @param copy: If False, data passed to fit are overwritten and running fit(X).transform(X) will
 * not yield the expected results, use fit_transform(X) instead.
 * @param whiten: When True (False by default) the components_ vectors are multiplied by the square
 * root of n_samples and then divided by the singular values to ensure uncorrelated outputs with
 * unit component-wise variances.
 * @param algorithm: the solver to be used in PCA.
 * @param tol: Tolerance for singular values computed by the Jacobi solver
 * @param n_iterations: Number of iterations for the power method computed by the Jacobi solver
 * @param verbose: 0: no error message printing, 1: print error messages
 */

template <typename enum_solver = solver>
class paramsPCATemplate : public paramsTSVDTemplate<enum_solver> {
 public:
  bool copy   = true;  // TODO unused, see #2830 and #2833
  bool whiten = false;
};

typedef paramsTSVDTemplate<> paramsTSVD;
typedef paramsPCATemplate<> paramsPCA;

typedef paramsPCATemplate<mg_solver> paramsPCAMG;
typedef paramsTSVDTemplate<mg_solver> paramsTSVDMG;

};  // end namespace ML
