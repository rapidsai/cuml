/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
