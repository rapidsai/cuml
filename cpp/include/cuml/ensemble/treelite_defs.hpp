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

#pragma once

// Same definition as ModelHandle in treelite, to avoid dependencies
// of cuML C++ headers on treelite headers.
// Original definition here:
// https://github.com/dmlc/treelite/blob/fca738770d2b09be1c0842fac9c0f5e3f6126c40/include/treelite/c_api.h#L25
typedef void* ModelHandle;
