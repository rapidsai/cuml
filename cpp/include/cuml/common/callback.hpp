/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
