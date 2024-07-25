/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include "glm_base.cuh"
#include "simple_mat.cuh"

#include <raft/linalg/add.cuh>
#include <raft/util/cuda_utils.cuh>

namespace ML {
namespace GLM {
namespace detail {

template <typename T>
struct SVCL1Loss : GLMBase<T, SVCL1Loss<T>> {
  typedef GLMBase<T, SVCL1Loss<T>> Super;

  const struct Lz {
    inline __device__ T operator()(const T y, const T z) const
    {
      T s = 2 * y - 1;
      return raft::max<T>(0, 1 - s * z);
    }
  } lz;

  const struct Dlz {
    inline __device__ T operator()(const T y, const T z) const
    {
      T s = 2 * y - 1;
      return s * z <= 1 ? -s : 0;
    }
  } dlz;

  SVCL1Loss(const raft::handle_t& handle, int D, bool has_bias)
    : Super(handle, D, 1, has_bias), lz{}, dlz{}
  {
  }

  inline T gradNorm(const SimpleVec<T>& grad, T* dev_scalar, cudaStream_t stream)
  {
    return nrm1(grad, dev_scalar, stream);
  }
};

template <typename T>
struct SVCL2Loss : GLMBase<T, SVCL2Loss<T>> {
  typedef GLMBase<T, SVCL2Loss<T>> Super;

  const struct Lz {
    inline __device__ T operator()(const T y, const T z) const
    {
      T s = 2 * y - 1;
      T t = raft::max<T>(0, 1 - s * z);
      return t * t;
    }
  } lz;

  const struct Dlz {
    inline __device__ T operator()(const T y, const T z) const
    {
      T s = 2 * y - 1;
      return s * z <= 1 ? z - s : 0;
    }
  } dlz;

  SVCL2Loss(const raft::handle_t& handle, int D, bool has_bias)
    : Super(handle, D, 1, has_bias), lz{}, dlz{}
  {
  }

  inline T gradNorm(const SimpleVec<T>& grad, T* dev_scalar, cudaStream_t stream)
  {
    return squaredNorm(grad, dev_scalar, stream) * 0.5;
  }
};

template <typename T>
struct SVRL1Loss : GLMBase<T, SVRL1Loss<T>> {
  typedef GLMBase<T, SVRL1Loss<T>> Super;

  const struct Lz {
    T sensitivity;
    inline __device__ T operator()(const T y, const T z) const
    {
      T t = y - z;
      return t > sensitivity ? t - sensitivity : t < -sensitivity ? -t - sensitivity : 0;
    }
  } lz;

  const struct Dlz {
    T sensitivity;
    inline __device__ T operator()(const T y, const T z) const
    {
      T t = y - z;
      return t > sensitivity ? -1 : (t < -sensitivity ? 1 : 0);
    }
  } dlz;

  SVRL1Loss(const raft::handle_t& handle, int D, bool has_bias, T sensitivity)
    : Super(handle, D, 1, has_bias), lz{sensitivity}, dlz{sensitivity}
  {
  }

  inline T gradNorm(const SimpleVec<T>& grad, T* dev_scalar, cudaStream_t stream)
  {
    return nrm1(grad, dev_scalar, stream);
  }
};

template <typename T>
struct SVRL2Loss : GLMBase<T, SVRL2Loss<T>> {
  typedef GLMBase<T, SVRL2Loss<T>> Super;

  const struct Lz {
    T sensitivity;
    inline __device__ T operator()(const T y, const T z) const
    {
      T t = y - z;
      T s = t > sensitivity ? t - sensitivity : t < -sensitivity ? -t - sensitivity : 0;
      return s * s;
    }
  } lz;

  const struct Dlz {
    T sensitivity;
    inline __device__ T operator()(const T y, const T z) const
    {
      T t = y - z;
      return -2 * (t > sensitivity ? t - sensitivity : t < -sensitivity ? (t + sensitivity) : 0);
    }
  } dlz;

  SVRL2Loss(const raft::handle_t& handle, int D, bool has_bias, T sensitivity)
    : Super(handle, D, 1, has_bias), lz{sensitivity}, dlz{sensitivity}
  {
  }

  inline T gradNorm(const SimpleVec<T>& grad, T* dev_scalar, cudaStream_t stream)
  {
    return squaredNorm(grad, dev_scalar, stream) * 0.5;
  }
};
};  // namespace detail
};  // namespace GLM
};  // namespace ML
