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

#include <raft/cudart_utils.h>
#include <rmm/mr/device/per_device_resource.hpp>

#include <vector>

namespace ML {

template <typename DataT, int Dim, typename IndexT = int>
class Tensor {
 public:
  enum { NumDim = Dim };
  typedef DataT* DataPtrT;

  __host__ ~Tensor()
  {
    if (_state == AllocState::Owner) {
      if (memory_type(_data) == cudaMemoryTypeHost) { delete _data; }

      if (memory_type(_data) == cudaMemoryTypeDevice) {
        rmm_alloc->deallocate(_data, this->getSizeInBytes(), _stream);
      } else if (memory_type(_data) == cudaMemoryTypeHost) {
        delete _data;
      }
    }
  }

  __host__ Tensor(DataPtrT data, const std::vector<IndexT>& sizes)
    : _data(data), _state(AllocState::NotOwner)
  {
    static_assert(Dim > 0, "must have > 0 dimensions");

    ASSERT(sizes.size() == Dim,
           "invalid argument: # of entries in the input argument 'sizes' must "
           "match the tensor dimension");

    for (int i = 0; i < Dim; ++i) {
      _size[i] = sizes[i];
    }

    _stride[Dim - 1] = (IndexT)1;
    for (int j = Dim - 2; j >= 0; --j) {
      _stride[j] = _stride[j + 1] * _size[j + 1];
    }
  }

  // allocate the data using the allocator and release when the object goes out of scope
  // allocating tensor is the owner of the data
  __host__ Tensor(const std::vector<IndexT>& sizes, cudaStream_t stream)
    : _stream(stream), _state(AllocState::Owner)
  {
    static_assert(Dim > 0, "must have > 0 dimensions");

    ASSERT(sizes.size() == Dim, "dimension mismatch");

    for (int i = 0; i < Dim; ++i) {
      _size[i] = sizes[i];
    }

    _stride[Dim - 1] = (IndexT)1;
    for (int j = Dim - 2; j >= 0; --j) {
      _stride[j] = _stride[j + 1] * _size[j + 1];
    }

    rmm_alloc = rmm::mr::get_current_device_resource();
    _data     = (DataT*)rmm_alloc->allocate(this->getSizeInBytes(), _stream);

    ASSERT(this->data() || (this->getSizeInBytes() == 0), "device allocation failed");
  }

  /// returns the total number of elements contained within our data
  __host__ size_t numElements() const
  {
    size_t num = (size_t)getSize(0);

    for (int i = 1; i < Dim; ++i) {
      num *= (size_t)getSize(i);
    }

    return num;
  }

  /// returns the size of a given dimension, `[0, Dim - 1]`
  __host__ inline IndexT getSize(int i) const { return _size[i]; }

  /// returns the stride array
  __host__ inline const IndexT* strides() const { return _stride; }

  /// returns the stride array.
  __host__ inline const IndexT getStride(int i) const { return _stride[i]; }

  /// returns the total size in bytes of our data
  __host__ size_t getSizeInBytes() const { return numElements() * sizeof(DataT); }

  /// returns a raw pointer to the start of our data
  __host__ inline DataPtrT data() { return _data; }

  /// returns a raw pointer to the start of our data.
  __host__ inline DataPtrT begin() { return _data; }

  /// returns a raw pointer to the end of our data
  __host__ inline DataPtrT end() { return data() + numElements(); }

  /// returns a raw pointer to the start of our data (const)
  __host__ inline DataPtrT data() const { return _data; }

  /// returns a raw pointer to the end of our data (const)
  __host__ inline DataPtrT end() const { return data() + numElements(); }

  /// returns the size array.
  __host__ inline const IndexT* sizes() const { return _size; }

  template <int NewDim>
  __host__ Tensor<DataT, NewDim, IndexT> view(const std::vector<IndexT>& sizes,
                                              const std::vector<IndexT>& start_pos)
  {
    ASSERT(sizes.size() == NewDim, "invalid view requested");
    ASSERT(start_pos.size() == Dim, "dimensionality of the position if incorrect");

    // calc offset at start_pos
    uint32_t offset = 0;
    for (uint32_t dim = 0; dim < Dim; ++dim) {
      offset += start_pos[dim] * getStride(dim);
    }
    DataPtrT newData = this->data() + offset;

    // The total size of the new view must be the <= total size of the old view
    size_t curSize = numElements();
    size_t newSize = 1;

    for (auto s : sizes) {
      newSize *= s;
    }

    ASSERT(newSize <= curSize, "invalid view requested");

    return Tensor<DataT, NewDim, IndexT>(newData, sizes);
  }

 private:
  enum AllocState {
    /// This tensor itself owns the memory, which must be freed via
    /// cudaFree
    Owner,

    /// This tensor itself is not an owner of the memory; there is
    /// nothing to free
    NotOwner
  };

 protected:
  /// Raw pointer to where the tensor data begins
  DataPtrT _data{};

  /// Array of strides (in sizeof(T) terms) per each dimension
  IndexT _stride[Dim];

  /// Size per each dimension
  IndexT _size[Dim];

  AllocState _state{};

  cudaStream_t _stream{};

  rmm::mr::device_memory_resource* rmm_alloc;
};

};  // end namespace ML
