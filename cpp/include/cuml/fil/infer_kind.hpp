/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuml/common/export.hpp>
namespace CUML_EXPORT ML {
namespace fil {
enum class infer_kind : unsigned char { default_kind = 0, per_tree = 1, leaf_id = 2 };
}
}  // namespace CUML_EXPORT ML
