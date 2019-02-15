/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "solver/solver_c.h"
#include "solver/learning_rate.h"
#include "functions/linearReg.h"
#include "linalg/binary_op.h"
#include "linalg/unary_op.h"
#include "matrix/math.h"

namespace UMAPAlgo {

    using namespace ML;

    namespace Optimize {

        /**
         * Calculate the gradients for training the embeddings in UMAP.
         * Rather The difference in this gradient descent is that
         * the parameters being updated are the embeddings themselves.
         *
         * Will need to think of a good way to incorporate this into
         * our SGD prim.
         */
        template<typename T>
        void umapEmbeddingLossGrads(T *input, int n_rows, int n_cols,
                const T *labels, const T *coef, T *grads, penalty pen,
                T alpha, T l1_ratio, cublasHandle_t cublas_handle) {

            // For the standard sampling:
            // Gradient is: -2.0 * a * b * pow(dist_squared, b - 1.0) / (a * pow(dist_squared, b) + 1.0)
            // For each d in the current: current_d += grad_d * learning_rate
            // if move_other: other_d += -grad_d * learning_rate

            // For the negative sampling:
            // gradient is: (2.0 * gamma * b) / ((0.001 + dist_squared) * (a * pow(dist_squared, b) + 1))
        }

        /**
         * This works on a single dimensional set of
         * x-values.
         */
        template<typename T>
        void f(const T *input, int n_rows,
               T *coef, int n_coeffs, T *preds,
               UMAPParams *params) {

            // Function: 1/1+ax^(2b)
            MLCommon::LinAlg::multiplyScalar(preds, input, params->a, n_rows);
            MLCommon::LinAlg::powerScalar(preds, preds, 2*params->b, n_rows);
            MLCommon::LinAlg::addScalar(preds, preds, 1, n_rows);
            MLCommon::Matrix::reciprocal(preds, n_rows);
        }

        /**
         * Calculate the gradients for fitting parameters a and b
         * to a smooth function based on exponential decay
         */
        template<typename T>
        void abLossGrads(T *input, int n_rows, int n_cols,
                const T *labels, const T *coef, n_coefs, T *grads, penalty pen,
                T alpha, T l1_ratio, cublasHandle_t cublas_handle) {

            UMAPParams *params; // todo: Need to be able to use the params
                                // associated w/ the UMAP class

            /**
             * Calculate f(x, a) for all a in 1..N
             */
            T *labels_pred;
            MLCommon::allocate(labels_pred, n_rows);

            f(input, n_rows, coef, n_coefs, labels_pred, params);

            /**
             * Calculate MSE
             */
            MLCommon::LinAlg::subtract(labels_pred, labels_pred, labels, n_rows);

            /**
             * Gradient w/ respect to a
             */
            T *a_deriv;
            MLCommon::copy(a_deriv, input, n_rows);

            // sum_error * (x^(2b)) / ((1+ax^(2b))^2)



            /**
             * Gradient w/ respect to b
             */
            T *b_deriv;
            MLCommon::copy(b_deriv, input, n_rows);

            // sum_error * -(2ax^(2b)*ln(x))/(1 + ax^(2b))^2



            /**
             * Finally, take the mean
             */
            MLCommon::Stats::mean(grads, input, n_cols, n_rows, false, false);

        }
    }
}

