/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cuml/linear_model/glm_api.h>
#include <cuml/linear_model/qn.h>

void test_glm()
{
  cumlHandle_t handle  = 0;
  cumlError_t response = CUML_SUCCESS;
  qn_params pams       = {.loss                = QN_LOSS_UNKNOWN,
                          .penalty_l1          = 0,
                          .penalty_l2          = 1.0,
                          .grad_tol            = 1e-4,
                          .change_tol          = 1e-5,
                          .max_iter            = 1000,
                          .linesearch_max_iter = 50,
                          .lbfgs_memory        = 5,
                          .verbose             = 0,
                          .fit_intercept       = true,
                          .penalty_normalized  = true};

  // Checking return type at compile time.
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  response = cumlSpQnFit(handle, &pams, NULL, NULL, 0, 1, 2, NULL, NULL, NULL, true);

  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  response = cumlDpQnFit(handle, &pams, NULL, NULL, 0, 1, 2, NULL, NULL, NULL, true);
}
