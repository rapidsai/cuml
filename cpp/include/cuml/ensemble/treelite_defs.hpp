/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

// Same definition as TreeliteModelHandle in treelite, to avoid dependencies
// of cuML C++ headers on treelite headers.
// Original definition here:
// https://github.com/dmlc/treelite/blob/6ca4eb5e699aa73d3721638fc1a3a43bf658a48b/include/treelite/c_api.h#L38
typedef void* TreeliteModelHandle;
