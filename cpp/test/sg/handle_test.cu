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

#include <gtest/gtest.h>

#include "cuML_api.h"

TEST(HandleTest, CreateHandleAndDestroy)
{
  cumlHandle_t handle;
  cumlError_t status = cumlCreate(&handle);
  EXPECT_EQ(CUML_SUCCESS, status);

  status = cumlDestroy(handle);
  EXPECT_EQ(CUML_SUCCESS, status);
}

TEST(HandleTest, DoubleDestoryFails)
{
  cumlHandle_t handle;
  cumlError_t status = cumlCreate(&handle);
  EXPECT_EQ(CUML_SUCCESS, status);

  status = cumlDestroy(handle);
  EXPECT_EQ(CUML_SUCCESS, status);
  // handle is destroyed
  status = cumlDestroy(handle);
  EXPECT_EQ(CUML_INVALID_HANDLE, status);
}

TEST(HandleTest, SetStream)
{
  cumlHandle_t handle;
  cumlError_t status = cumlCreate(&handle);
  EXPECT_EQ(CUML_SUCCESS, status);

  status = cumlSetStream(handle, 0);
  EXPECT_EQ(CUML_SUCCESS, status);

  status = cumlDestroy(handle);
  EXPECT_EQ(CUML_SUCCESS, status);
}

TEST(HandleTest, SetStreamInvalidHandle)
{
  cumlHandle_t handle = 12346;
  EXPECT_EQ(CUML_INVALID_HANDLE, cumlSetStream(handle, 0));
}
