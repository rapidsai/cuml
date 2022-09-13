#pragma once
#include <cstddef>
#ifndef __CUDACC__
#include <math.h>
#endif
#include <variant>
#include <stddef.h>
#include <type_traits>
#include <variant>
#include <kayak/device_type.hpp>
#include <kayak/gpu_support.hpp>

namespace kayak {
template<typename index_t=size_t, typename storage_t=std::byte>
struct bitset {
  using storage_type = storage_t;
  using index_type = index_t;

  auto constexpr static const bin_width = index_type(
    sizeof(storage_type) * 8
  );

  HOST DEVICE bitset()
    : data_{nullptr}, num_bits_{0}
  {
  }

  HOST DEVICE bitset(storage_type* data, index_type size)
    : data_{data}, num_bits_{size}
  {
  }

  HOST DEVICE bitset(storage_type* data)
    : data_{data}, num_bits_(sizeof(storage_type) * 8)
  {
  }

  HOST DEVICE auto size() const {
    return num_bits_;
  }
  HOST DEVICE auto bin_count() const {
    return num_bits_ / bin_width + (num_bits_ % bin_width != 0);
  }

  // Standard bit-wise mutators and accessor
  HOST DEVICE auto& set(index_type index) {
    data_[bin_from_index(index)] |= mask_in_bin(index);
    return *this;
  }
  HOST DEVICE auto& clear(index_type index) {
    data_[bin_from_index(index)] &= ~mask_in_bin(index);
    return *this;
  }
  HOST DEVICE auto test(index_type index) const {
    auto result = false;
    if (index < num_bits_) {
      result = ((data_[bin_from_index(index)] & mask_in_bin(index)) != 0);
    }
    return result;
  }
  HOST DEVICE auto& flip() {
    for (auto i = index_type{}; i < bin_count(); ++i) {
      data_[i] = ~data_[i];
    }
    return *this;
  }

  // Bit-wise boolean operations
  HOST DEVICE auto& operator&=(bitset<storage_type> const& other) {
    for (auto i = index_type{}; i < min(size(), other.size()); ++i) {
      data_[i] &= other.data_[i];
    }
    return *this;
  }
  HOST DEVICE auto& operator|=(bitset<storage_type> const& other) {
    for (auto i = index_type{}; i < min(size(), other.size()); ++i) {
      data_[i] |= other.data_[i];
    }
    return *this;
  }
  HOST DEVICE auto& operator^=(bitset<storage_type> const& other) {
    for (auto i = index_type{}; i < min(size(), other.size()); ++i) {
      data_[i] ^= other.data_[i];
    }
    return *this;
  }
  HOST DEVICE auto& operator~() const {
    flip();
    return *this;
  }

 private:
  storage_type* data_;
  index_type num_bits_;

  HOST DEVICE auto mask_in_bin(index_type index) const {
    return storage_type{1} << (index % bin_width);
  }

  HOST DEVICE auto bin_from_index(index_type index) const {
    return index / bin_width;
  }
};

}
