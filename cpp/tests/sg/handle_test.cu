/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/cuml_api.h>

#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

TEST(HandleTest, CreateHandleAndDestroy)
{
  cumlHandle_t handle;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cumlError_t status = cumlCreate(&handle, stream);
  EXPECT_EQ(CUML_SUCCESS, status);

  status = cumlDestroy(handle);
  EXPECT_EQ(CUML_SUCCESS, status);
}

TEST(HandleTest, DoubleDestoryFails)
{
  cumlHandle_t handle;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cumlError_t status = cumlCreate(&handle, stream);
  EXPECT_EQ(CUML_SUCCESS, status);

  status = cumlDestroy(handle);
  EXPECT_EQ(CUML_SUCCESS, status);
  // handle is destroyed
  status = cumlDestroy(handle);
  EXPECT_EQ(CUML_INVALID_HANDLE, status);
}
