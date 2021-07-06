/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

namespace ML {
namespace Internals {

class Callback {
 public:
  virtual ~Callback() {}
};

class GraphBasedDimRedCallback : public Callback {
 public:
  template <typename T>
  void setup(int n, int n_components)
  {
    this->n            = n;
    this->n_components = n_components;
    this->isFloat      = std::is_same<T, float>::value;
  }

  virtual void on_preprocess_end(void* embeddings) = 0;
  virtual void on_epoch_end(void* embeddings)      = 0;
  virtual void on_train_end(void* embeddings)      = 0;

 protected:
  int n;
  int n_components;
  bool isFloat;
};

}  // namespace Internals
}  // namespace ML
