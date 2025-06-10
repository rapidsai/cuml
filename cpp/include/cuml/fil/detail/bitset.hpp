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
#include <cuml/fil/detail/raft_proto/device_type.hpp>
#include <cuml/fil/detail/raft_proto/gpu_support.hpp>

#include <cstddef>
#include <type_traits>
#include <variant>

#ifndef __CUDACC__
#include <math.h>
#endif

namespace ML {
namespace fil {
namespace detail {
template <typename index_t = size_t, typename storage_t = std::byte>
struct bitset {
  using storage_type = storage_t;
  using index_type   = index_t;

  auto constexpr static const bin_width = index_type(sizeof(storage_type) * 8);

  HOST DEVICE bitset() : data_{nullptr}, num_bits_{0} {}

  HOST DEVICE bitset(storage_type* data, index_type size) : data_{data}, num_bits_{size} {}

  HOST DEVICE bitset(storage_type* data) : data_{data}, num_bits_(sizeof(storage_type) * 8) {}

  HOST DEVICE auto size() const { return num_bits_; }
  HOST DEVICE auto bin_count() const
  {
    return num_bits_ / bin_width + (num_bits_ % bin_width != 0);
  }

  // Standard bit-wise mutators and accessor
  HOST DEVICE auto& set(index_type index)
  {
    data_[bin_from_index(index)] |= mask_in_bin(index);
    return *this;
  }
  HOST DEVICE auto& clear(index_type index)
  {
    data_[bin_from_index(index)] &= ~mask_in_bin(index);
    return *this;
  }
  HOST DEVICE auto test(index_type index) const
  {
    auto result = false;
    if (index < num_bits_) { result = ((data_[bin_from_index(index)] & mask_in_bin(index)) != 0); }
    return result;
  }
  HOST DEVICE auto& flip()
  {
    for (auto i = index_type{}; i < bin_count(); ++i) {
      data_[i] = ~data_[i];
    }
    return *this;
  }

  // Bit-wise boolean operations
  HOST DEVICE auto& operator&=(bitset<storage_type> const& other)
  {
    for (auto i = index_type{}; i < min(size(), other.size()); ++i) {
      data_[i] &= other.data_[i];
    }
    return *this;
  }
  HOST DEVICE auto& operator|=(bitset<storage_type> const& other)
  {
    for (auto i = index_type{}; i < min(size(), other.size()); ++i) {
      data_[i] |= other.data_[i];
    }
    return *this;
  }
  HOST DEVICE auto& operator^=(bitset<storage_type> const& other)
  {
    for (auto i = index_type{}; i < min(size(), other.size()); ++i) {
      data_[i] ^= other.data_[i];
    }
    return *this;
  }
  HOST DEVICE auto& operator~() const
  {
    flip();
    return *this;
  }

 private:
  storage_type* data_;
  index_type num_bits_;

  HOST DEVICE auto mask_in_bin(index_type index) const
  {
    return storage_type{1} << (index % bin_width);
  }

  HOST DEVICE auto bin_from_index(index_type index) const { return index / bin_width; }
};

}  // namespace detail
}  // namespace fil
}  // namespace ML
