/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/cluster/dbscan_api.h>

void test_dbscan()
{
  cumlHandle_t handle  = 0;
  cumlError_t response = CUML_SUCCESS;

  // Checking return type at compile time.
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  response = cumlSpDbscanFit(handle, NULL, 0, 1, 1.0f, 2, NULL, NULL, 10, 1);

  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  response = cumlDpDbscanFit(handle, NULL, 0, 1, 1.0, 2, NULL, NULL, 10, 1);
}
