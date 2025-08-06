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
#include <exception>
#include <string>

namespace ML {
namespace fil {

/** Exception indicating model is incompatible with FIL */
struct unusable_model_exception : std::exception {
  unusable_model_exception() : msg_{"Model is not compatible with FIL"} {}
  unusable_model_exception(std::string msg) : msg_{msg} {}
  unusable_model_exception(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_.c_str(); }

 private:
  std::string msg_;
};

/** Exception indicating model import failed */
struct model_import_error : std::exception {
  model_import_error() : model_import_error("Error while importing model") {}
  model_import_error(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

/**
 * Exception indicating a mismatch between the type of input data and the
 * model
 *
 * This typically occurs when doubles are provided as input to a model with
 * float thresholds or vice versa.
 */
struct type_error : std::exception {
  type_error() : type_error("Model cannot be used with given data type") {}
  type_error(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

}  // namespace fil
}  // namespace ML
