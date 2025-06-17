/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
namespace ML {
namespace fil {

/** Enum representing possible row-wise operations on output */
enum struct row_op : unsigned char {
  disable   = 0b00100000,
  softmax   = 0b01000000,
  max_index = 0b10000000
};

/** Enum representing possible element-wise operations on output */
enum struct element_op : unsigned char {
  disable                = 0b00000000,
  signed_square          = 0b00000001,
  hinge                  = 0b00000010,
  sigmoid                = 0b00000100,
  exponential            = 0b00001000,
  logarithm_one_plus_exp = 0b00010000
};

}  // namespace fil
}  // namespace ML
