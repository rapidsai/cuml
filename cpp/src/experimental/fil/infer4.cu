/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <cuml/experimental/fil/detail/device_initialization/gpu.cuh>
#include <cuml/experimental/fil/detail/specializations/device_initialization_macros.hpp>
#include <cuml/experimental/fil/detail/infer/gpu.cuh>
#include <cuml/experimental/fil/detail/specializations/infer_macros.hpp>
namespace ML {
namespace experimental {
namespace fil {
namespace detail {
namespace inference {
CUML_FIL_INFER_ALL(template, raft_proto::device_type::gpu, 4)
}
namespace device_initialization {
CUML_FIL_INITIALIZE_DEVICE(template, 4)
}
}
}
}
}
