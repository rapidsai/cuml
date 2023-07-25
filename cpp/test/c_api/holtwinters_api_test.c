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

#include <cuml/tsa/holtwinters_api.h>

void test_holtwinters()
{
  cumlHandle_t handle  = 0;
  cumlError_t response = CUML_SUCCESS;

  // Checking return type at compile time.
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  response = cumlHoltWinters_buffer_size(0, 1, 2, NULL, NULL, NULL, NULL, NULL, NULL);

  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  response =
    cumlHoltWintersSp_fit(handle, 0, 1, 2, 3, ADDITIVE, 1.0f, NULL, NULL, NULL, NULL, NULL);

  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  response =
    cumlHoltWintersDp_fit(handle, 0, 1, 2, 3, ADDITIVE, 1.0f, NULL, NULL, NULL, NULL, NULL);

  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  response = cumlHoltWintersSp_forecast(handle, 0, 1, 2, 3, ADDITIVE, NULL, NULL, NULL, NULL);

  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  response = cumlHoltWintersDp_forecast(handle, 0, 1, 2, 3, ADDITIVE, NULL, NULL, NULL, NULL);
}
