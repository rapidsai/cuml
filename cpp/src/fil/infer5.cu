/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuml/fil/detail/device_initialization/gpu.cuh>
#include <cuml/fil/detail/infer/gpu.cuh>
#include <cuml/fil/detail/specializations/device_initialization_macros.hpp>
#include <cuml/fil/detail/specializations/infer_macros.hpp>
namespace ML {
namespace fil {
namespace detail {
namespace inference {
CUML_FIL_INFER_ALL(template, raft_proto::device_type::gpu, 5)
}
namespace device_initialization {
CUML_FIL_INITIALIZE_DEVICE(template, 5)
}
}  // namespace detail
}  // namespace fil
}  // namespace ML
