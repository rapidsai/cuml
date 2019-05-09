/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

namespace ML {

/**
 * @defgroup pcaSolver: enumeration for pca solvers.
 * @param AUTO: Fastest solver will be used based on input shape and n_components.
 * @param FULL: All the eigenvectors and singular values (or eigenvalues) will be generated.
 * @param ARPACK: tsvd using power method. Lanczos will be included in the future.
 * @param RANDOMIZED: randomized svd
 * @param COV_EIG_DQ: covariance of input will be used along with eigen decomposition using divide and conquer method for symmetric matrices
 * @param COV_EIG_JACOBI: covariance of input will be used along with eigen decomposition using jacobi method for symmetric matrices
 * @{
 */
enum solver {
	COV_EIG_DQ, COV_EIG_JACOBI, RANDOMIZED,
};

enum lr_type {
	OPTIMAL, CONSTANT, INVSCALING, ADAPTIVE,
};

enum loss_funct {
	SQRD_LOSS, HINGE, LOG,
};

enum penalty {
	NONE, L1, L2, ELASTICNET
};

//template<typename math_t>
class params {
public:
	int n_rows;
	int n_cols;
	int gpu_id = 0;
};

//template<typename math_t>
//class paramsSolver: public params<math_t> {
class paramsSolver: public params{
public:
	int n_rows;
	int n_cols;
	//math_t tol = 0.0;
	float tol = 0.0;
    int n_iterations = 15;
	int random_state;
	int verbose = 0;
};

//template<typename math_t>
//class paramsTSVD: public paramsSolver<math_t> {
class paramsTSVD: public paramsSolver {
public:
	int n_components = 1;
	int max_sweeps = 15;
	solver algorithm = solver::COV_EIG_DQ;
	bool trans_input = false;
};

/**
 * @defgroup paramsPCA: structure for pca parameters. Ref: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
 * @param n_components: Number of components to keep. if n_components is not set all components are kept:
 * @param copy: If False, data passed to fit are overwritten and running fit(X).transform(X) will not yield the expected results,
 *              use fit_transform(X) instead.
 * @param whiten: When True (False by default) the components_ vectors are multiplied by the square root of n_samples and
 *                then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances.
 * @param svd_solver: the solver to be used in PCA.
 * @param tol: Tolerance for singular values computed by svd_solver == ‘arpack’ or svd_solver == ‘COV_EIG_JACOBI’
 * @param iterated_power: Number of iterations for the power method computed by svd_solver == ‘randomized’ or
 *                        jacobi method by svd_solver == 'COV_EIG_JACOBI'.
 * @random_state: RandomState instance or None, optional (default None)
 * @verbose: 0: no error message printing, 1: print error messages
 * @max_sweeps: number of sweeps jacobi method uses. The more the better accuracy.
 * @{
 */

//template<typename math_t>
//class paramsPCA: public paramsTSVD<math_t> {
class paramsPCA: public paramsTSVD {
public:
	bool copy = true;
	bool whiten = false;
};

}; // end namespace ML
