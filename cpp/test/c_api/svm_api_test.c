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

#include <cuml/svm/svm_api.h>

void test_svm()
{
  cumlHandle_t handle  = 0;
  cumlError_t response = CUML_SUCCESS;

  // Checking return type at compile time.
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  response = cumlSpSvcFit(handle,
                          NULL,
                          0,
                          1,
                          NULL,
                          1.0f,
                          2.0f,
                          2,
                          3,
                          3.0f,
                          4,
                          LINEAR,
                          5,
                          6.0f,
                          7.0f,
                          NULL,
                          NULL,
                          NULL,
                          NULL,
                          NULL,
                          NULL,
                          NULL);

  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  response = cumlDpSvcFit(handle,
                          NULL,
                          0,
                          1,
                          NULL,
                          1.0,
                          2.0,
                          2,
                          3,
                          3.0,
                          4,
                          LINEAR,
                          5,
                          6.0,
                          7.0,
                          NULL,
                          NULL,
                          NULL,
                          NULL,
                          NULL,
                          NULL,
                          NULL);

  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  response = cumlSpSvcPredict(
    handle, NULL, 0, 1, LINEAR, 2, 3.0f, 4.0f, 5, 6.0f, NULL, NULL, 7, NULL, NULL, 8.0f, 9);

  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  response = cumlDpSvcPredict(
    handle, NULL, 0, 1, LINEAR, 2, 3.0, 4.0, 5, 6.0, NULL, NULL, 7, NULL, NULL, 8.0, 9);
}
