/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include <raft/cuda_utils.cuh>
#include <raft/linalg/binary_op.cuh>
#include "glm_base.cuh"
#include "simple_mat.cuh"

namespace ML {
namespace GLM {

template <typename T>
struct SVCL1Loss : GLMBase<T, SVCL1Loss<T>> {
  typedef GLMBase<T, SVCL1Loss<T>> Super;

  SVCL1Loss(const raft::handle_t& handle, int D, bool has_bias) : Super(handle, D, 1, has_bias) {}
  inline __device__ T lz(const T y, const T z) const { return raft::myMax<T>(0, 1 - y * z); }
  inline __device__ T dlz(const T y, const T z) const { return y * z <= 1 ? -y : 0; }
};

template <typename T>
struct SVCL2Loss : GLMBase<T, SVCL2Loss<T>> {
  typedef GLMBase<T, SVCL2Loss<T>> Super;

  SVCL2Loss(const raft::handle_t& handle, int D, bool has_bias) : Super(handle, D, 1, has_bias) {}
  inline __device__ T lz(const T y, const T z) const
  {
    T t = raft::myMax<T>(0, 1 - y * z);
    return t * t;
  }
  inline __device__ T dlz(const T y, const T z) const { return y * z <= 1 ? z - y : 0; }
};

template <typename T>
struct SVRL1Loss : GLMBase<T, SVRL1Loss<T>> {
  typedef GLMBase<T, SVRL1Loss<T>> Super;

  struct Lz {
    T sensitivity;
    inline __device__ T operator()(const T y, const T z) const
    {
      T t = y - z;
      return t > sensitivity ? t - sensitivity : t < -sensitivity ? -t - sensitivity : 0;
    }
  };

  struct Dlz {
    T sensitivity;
    inline __device__ T operator()(const T y, const T z) const
    {
      T t = y - z;
      return t > sensitivity ? -1 : (t < -sensitivity ? 1 : 0);
    }
  };

  const Lz lz;
  const Dlz dlz;

  SVRL1Loss(const raft::handle_t& handle, int D, bool has_bias, T sensitivity)
    : Super(handle, D, 1, has_bias), lz{sensitivity}, dlz{sensitivity}
  {
  }
};

template <typename T>
struct SVRL2Loss : GLMBase<T, SVRL2Loss<T>> {
  typedef GLMBase<T, SVRL2Loss<T>> Super;

  struct Lz {
    T sensitivity;
    inline __device__ T operator()(const T y, const T z) const
    {
      T t = y - z;
      T s = t > sensitivity ? t - sensitivity : t < -sensitivity ? -t - sensitivity : 0;
      return s * s;
    }
  };

  struct Dlz {
    T sensitivity;
    inline __device__ T operator()(const T y, const T z) const
    {
      T t = y - z;
      return -2 * (t > sensitivity ? t - sensitivity : t < -sensitivity ? (t + sensitivity) : 0);
    }
  };

  const Lz lz;
  const Dlz dlz;

  SVRL2Loss(const raft::handle_t& handle, int D, bool has_bias, T sensitivity)
    : Super(handle, D, 1, has_bias), lz{sensitivity}, dlz{sensitivity}
  {
  }
};

};  // namespace GLM
};  // namespace ML
