/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuml/fil/detail/infer/cpu.hpp>
#include <cuml/fil/detail/specializations/infer_macros.hpp>
namespace ML {
namespace fil {
namespace detail {
namespace inference {
CUML_FIL_INFER_ALL(template, raft_proto::device_type::cpu, 6)
}
}  // namespace detail
}  // namespace fil
}  // namespace ML
