/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
namespace ML {
namespace fil {
enum class infer_kind : unsigned char { default_kind = 0, per_tree = 1, leaf_id = 2 };
}
}  // namespace ML
