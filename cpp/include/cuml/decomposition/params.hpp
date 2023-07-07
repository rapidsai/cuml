/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>

namespace ML {

/**
 * @param COV_EIG_DQ: covariance of input will be used along with eigen decomposition using divide
 * and conquer method for symmetric matrices
 * @param COV_EIG_JACOBI: covariance of input will be used along with eigen decomposition using
 * jacobi method for symmetric matrices
 */
enum class solver : int {
  COV_EIG_DQ,
  COV_EIG_JACOBI,
};

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
 * @param tol: Tolerance for singular values computed by svd_solver == ‘arpack’ or svd_solver ==
 * ‘COV_EIG_JACOBI’
 * @param n_iterations: Number of iterations for the power method computed by jacobi method
 * (svd_solver == 'COV_EIG_JACOBI').
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

enum class mg_solver { COV_EIG_DQ, COV_EIG_JACOBI, QR };

typedef paramsPCATemplate<mg_solver> paramsPCAMG;
typedef paramsTSVDTemplate<mg_solver> paramsTSVDMG;

};  // end namespace ML
