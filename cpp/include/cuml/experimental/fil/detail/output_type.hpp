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
#pragma once

#include <type_traits>
#include <cstddef>

namespace ML {
namespace experimental {
namespace fil {
namespace detail {

template <typename forest_t, typename vector_output_t>
using output_t = std::conditional_t<
    !std::is_same_v<vector_output_t, std::nullptr_t>,
    std::remove_pointer_t<vector_output_t>,
    typename forest_t::node_type::threshold_type
>;

}
}
}
}
