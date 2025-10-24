/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/neighbors/knn_api.h>

void test_knn()
{
  cumlHandle_t handle  = 0;
  cumlError_t response = CUML_SUCCESS;

  // Checking return type at compile time.
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  response =
    knn_search(handle, NULL, NULL, 1, 2, NULL, 3, NULL, NULL, 4, true, false, 0, 2.0f, false);
}
