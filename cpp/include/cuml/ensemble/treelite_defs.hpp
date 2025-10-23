/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Same definition as TreeliteModelHandle in treelite, to avoid dependencies
// of cuML C++ headers on treelite headers.
// Original definition here:
// https://github.com/dmlc/treelite/blob/6ca4eb5e699aa73d3721638fc1a3a43bf658a48b/include/treelite/c_api.h#L38
typedef void* TreeliteModelHandle;
