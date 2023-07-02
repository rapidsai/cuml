/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cstdio>
#include <cstdlib>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>

#include <raft/core/handle.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace ML {
namespace Dbscan {
namespace Multigroups {
namespace Metadata {

const std::size_t align = 256;

template <typename Index_t = int>
__global__ void init_offset_mask(Index_t* mask,
                                 const Index_t* stride,
                                 const Index_t* position,
                                 Index_t n_groups)
{
  int group_id = blockIdx.x;
  if (group_id >= n_groups) return;
  Index_t* selected_mask = mask + position[group_id];
  for (int i = threadIdx.x; i < stride[group_id]; i += blockDim.x)
    selected_mask[i] = group_id;
  return;
}

/**
 * Metadata including the number of groups, numbers of rows / cols from different groups on both
 * host and device, and prefix sum of rows / cols on device.
 */
template <typename Index_t = int>
class MultiGroupMetaData {
 public:
  MultiGroupMetaData(Index_t _n_groups, Index_t* _n_rows_ptr, Index_t _n_cols)
    : n_groups(_n_groups), n_rows_ptr(_n_rows_ptr), is_const_cols(true)
  {
    n_cols_ptr = reinterpret_cast<Index_t*>(malloc(n_groups * sizeof(Index_t)));
    thrust::fill_n(thrust::host, n_cols_ptr, n_groups, _n_cols);
    max_rows = *thrust::max_element(thrust::host, n_rows_ptr, n_rows_ptr + n_groups);
    max_cols = _n_cols;
    sum_rows = thrust::reduce(thrust::host, n_rows_ptr, n_rows_ptr + n_groups);
    sum_cols = _n_cols * n_groups;
  }

  MultiGroupMetaData(Index_t _n_groups, Index_t* _n_rows_ptr, Index_t* _n_cols_ptr)
    : n_groups(_n_groups), n_rows_ptr(_n_rows_ptr), n_cols_ptr(_n_cols_ptr), is_const_cols(false)
  {
    max_rows = *thrust::max_element(thrust::host, n_rows_ptr, n_rows_ptr + n_groups);
    max_cols = *thrust::max_element(thrust::host, n_cols_ptr, n_cols_ptr + n_groups);
    sum_rows = thrust::reduce(thrust::host, n_rows_ptr, n_rows_ptr + n_groups);
    sum_cols = thrust::reduce(thrust::host, n_cols_ptr, n_cols_ptr + n_groups);
  }

  ~MultiGroupMetaData()
  {
    if (is_const_cols && n_cols_ptr != nullptr) {
      free(n_cols_ptr);
      n_cols_ptr = nullptr;
    }
  }

  size_t get_wsp_size()
  {
    size_t workspace_size = raft::alignTo<std::size_t>(sizeof(Index_t) * (n_groups + 1), align);
    workspace_size *= (is_const_cols) ? 2 : 4;
    return workspace_size;
  }

  void initialize(const raft::handle_t& handle,
                  void* _workspace,
                  size_t buffer_size,
                  cudaStream_t stream)
  {
    wsp_size = this->get_wsp_size();
    ASSERT(buffer_size == wsp_size,
           "The required size of workspace (%ld) doesn't match that passed (%ld) in %s.\n",
           wsp_size,
           buffer_size,
           __FUNCTION__);
    workspace         = _workspace;
    size_t chunk_size = wsp_size / ((is_const_cols) ? 2 : 4);
    char* wsp_ptr     = reinterpret_cast<char*>(_workspace);
    this->reset(stream);

    dev_n_rows = reinterpret_cast<Index_t*>(wsp_ptr);
    wsp_ptr += chunk_size;
    dev_pfxsum_rows = reinterpret_cast<Index_t*>(wsp_ptr);
    wsp_ptr += chunk_size;
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      dev_n_rows, n_rows_ptr, n_groups * sizeof(Index_t), cudaMemcpyHostToDevice, stream));
    thrust::device_ptr<Index_t> thrust_dev_array  = thrust::device_pointer_cast(dev_n_rows);
    thrust::device_ptr<Index_t> thrust_dev_pfxsum = thrust::device_pointer_cast(dev_pfxsum_rows);
    thrust::exclusive_scan(handle.get_thrust_policy(),
                           thrust_dev_array,
                           thrust_dev_array + (n_groups + 1),
                           thrust_dev_pfxsum);

    if (is_const_cols) {
      dev_n_cols = reinterpret_cast<Index_t*>(wsp_ptr);
      wsp_ptr += chunk_size;
      dev_pfxsum_cols   = reinterpret_cast<Index_t*>(wsp_ptr);
      thrust_dev_array  = thrust::device_pointer_cast(dev_n_cols);
      thrust_dev_pfxsum = thrust::device_pointer_cast(dev_pfxsum_cols);
      thrust::exclusive_scan(handle.get_thrust_policy(),
                             thrust_dev_array,
                             thrust_dev_array + (n_groups + 1),
                             thrust_dev_pfxsum);
    }
    return;
  }

  void destroy()
  {
    wsp_size        = 0;
    workspace       = nullptr;
    dev_n_rows      = nullptr;
    dev_n_cols      = nullptr;
    dev_pfxsum_rows = nullptr;
    dev_pfxsum_cols = nullptr;
    return;
  }

  HDI const Index_t* get_host_rows() const noexcept { return n_rows_ptr; }
  HDI const Index_t* get_host_cols() const noexcept { return n_cols_ptr; }
  HDI const Index_t* get_dev_rows() const noexcept { return dev_n_rows; }
  HDI const Index_t* get_dev_cols() const noexcept { return dev_n_cols; }
  HDI const Index_t* get_dev_pfxsum_rows() const noexcept { return dev_pfxsum_rows; }
  HDI const Index_t* get_dev_pfxsum_cols() const noexcept { return dev_pfxsum_cols; }

  const Index_t n_groups;
  Index_t max_rows;
  Index_t sum_rows;
  Index_t max_cols;
  Index_t sum_cols;
  const bool is_const_cols;

 private:
  void reset(cudaStream_t stream)
  {
    if (workspace != nullptr && wsp_size != 0) {
      RAFT_CUDA_TRY(cudaMemsetAsync(workspace, 0, wsp_size, stream));
    }
  }

  Index_t* n_rows_ptr      = nullptr;
  Index_t* n_cols_ptr      = nullptr;
  Index_t* dev_n_rows      = nullptr;
  Index_t* dev_n_cols      = nullptr;
  Index_t* dev_pfxsum_rows = nullptr;
  Index_t* dev_pfxsum_cols = nullptr;
  size_t wsp_size          = 0;
  void* workspace          = nullptr;
};

/**
 * Accessor is used to access data from concatenated matrixes.
 */
template <typename Data_t,
          typename Index_t       = int,
          typename MetaDataClass = MultiGroupMetaData<Index_t>>
class BaseAccessor {
 public:
  BaseAccessor(const MetaDataClass* _metadata, Data_t* _data)
    : metadata(_metadata),
      n_groups(_metadata->n_groups),
      n_points(_metadata->sum_rows),
      n_rows_ptr(_metadata->get_dev_rows()),
      row_start_ids(_metadata->get_dev_pfxsum_rows()),
      m_data(_data)
  {
  }
  BaseAccessor(const MetaDataClass* _metadata, const Data_t* _data)
    : metadata(_metadata),
      n_groups(_metadata->n_groups),
      n_points(_metadata->sum_rows),
      n_rows_ptr(_metadata->get_dev_rows()),
      row_start_ids(_metadata->get_dev_pfxsum_rows()),
      m_data(_data)
  {
  }

  virtual void initialize(const raft::handle_t& handle,
                          void* workspace,
                          size_t buffer_size,
                          cudaStream_t stream)
  {
  }
  virtual void destroy() {}
  virtual const Data_t* data() const noexcept { return m_data; }

  const MetaDataClass* metadata;
  const Index_t n_groups;
  const Index_t n_points;
  const Index_t* n_rows_ptr;
  const Index_t* row_start_ids;

 private:
  const Data_t* m_data;
};

template <typename Data_t,
          typename Index_t       = int,
          typename MetaDataClass = MultiGroupMetaData<Index_t>,
          typename BaseClass     = BaseAccessor<Data_t, Index_t, MetaDataClass>>
class PointAccessor : public BaseClass {
 public:
  PointAccessor(const MetaDataClass* _metadata, const Data_t* _data)
    : BaseClass(_metadata, _data),
      max_rows(_metadata->max_rows),
      feat_size(_metadata->max_cols),
      pts(_data)
  {
  }

  const Index_t max_rows;
  const Index_t feat_size;
  const Data_t* pts;
};

template <typename Data_t,
          typename Index_t       = int,
          typename MetaDataClass = MultiGroupMetaData<Index_t>,
          typename BaseClass     = BaseAccessor<Data_t, Index_t, MetaDataClass>>
class VertexDegAccessor : public BaseClass {
 public:
  VertexDegAccessor(const MetaDataClass* _metadata, Data_t* _data) : BaseClass(_metadata, _data)
  {
    Data_t* temp = _data;
    this->vd     = temp;
    temp += this->n_points;
    this->vd_all = temp;
    temp += 1;
    this->vd_group = temp;
  }

  Data_t* vd;
  Data_t* vd_group;
  Data_t* vd_all;
};

template <typename Data_t,
          typename Index_t       = int,
          typename MetaDataClass = MultiGroupMetaData<Index_t>,
          typename BaseClass     = BaseAccessor<Data_t, Index_t, MetaDataClass>>
class AdjGraphAccessor : public BaseClass {
 public:
  AdjGraphAccessor(const MetaDataClass* _metadata, Data_t* _data)
    : BaseClass(_metadata, _data), max_nbr(_metadata->max_rows), adj(_data)
  {
  }

  size_t get_wsp_size()
  {
    return raft::alignTo<std::size_t>(this->n_groups * sizeof(Index_t), align) +
           raft::alignTo<std::size_t>(this->n_groups * sizeof(std::size_t), align);
  }

  size_t get_layout_size()
  {
    size_t sta_layout_size =
      raft::alignTo<std::size_t>(sizeof(bool) * this->n_points * max_nbr, align);
    const Index_t* host_n_rows = this->metadata->get_host_rows();
    size_t dyn_layout_size =
      thrust::reduce(thrust::host,
                     host_n_rows,
                     host_n_rows + this->n_groups,
                     static_cast<size_t>(0),
                     [](const size_t& lhs, const Index_t& rhs) {
                       return lhs + raft::alignTo<std::size_t>(sizeof(bool) * rhs * rhs, align);
                     });

    optim_layout = dyn_layout_size < sta_layout_size;
    return (optim_layout) ? dyn_layout_size : sta_layout_size;
  }

  void initialize(const raft::handle_t& handle,
                  void* workspace,
                  size_t buffer_size,
                  cudaStream_t stream)
  {
    wsp_size = this->get_wsp_size();
    this->get_layout_size();
    ASSERT(buffer_size == wsp_size,
           "The required size of workspace (%ld) doesn't match that passed (%ld) in %s.\n",
           wsp_size,
           buffer_size,
           __FUNCTION__);

    if (adj_col_stride != nullptr || adj_group_offset != nullptr) { this->destroy(); }
    char* wsp_ptr    = reinterpret_cast<char*>(workspace);
    adj_col_stride   = reinterpret_cast<Index_t*>(wsp_ptr);
    adj_group_offset = reinterpret_cast<std::size_t*>(
      wsp_ptr + raft::alignTo<std::size_t>(this->n_groups * sizeof(Index_t), align));
    RAFT_CUDA_TRY(cudaMemcpyAsync(adj_col_stride,
                                  this->n_rows_ptr,
                                  this->n_groups * sizeof(Index_t),
                                  cudaMemcpyDeviceToDevice,
                                  stream));
    h_adj_group_offset = new std::size_t[this->n_groups];
    // cudaMallocHost(&h_adj_group_offset, this->n_groups * sizeof(Index_t));

    if (!optim_layout) {
      auto row_start_ids_view = raft::make_device_vector_view(this->row_start_ids, this->n_groups);
      auto adj_group_offset_view =
        raft::make_device_vector_view(const_cast<std::size_t*>(adj_group_offset), this->n_groups);
      raft::linalg::map(
        handle, adj_group_offset_view, raft::cast_op<std::size_t>{}, row_start_ids_view);

      auto offset_in_view = raft::make_device_vector_view(
        const_cast<const std::size_t*>(adj_group_offset), this->n_groups);
      auto offset_out_view =
        raft::make_device_vector_view(const_cast<std::size_t*>(adj_group_offset), this->n_groups);
      raft::linalg::map(handle,
                        offset_out_view,
                        raft::mul_const_op<Index_t>(static_cast<Index_t>(max_nbr)),
                        offset_in_view);
    } else {
      const Index_t* host_n_rows = this->metadata->get_host_rows();
      for (struct {
             int i;
             std::size_t offset;
           } v = {0, 0};
           v.i < this->n_groups;
           ++v.i) {
        h_adj_group_offset[v.i] = v.offset;
        v.offset +=
          raft::alignTo<std::size_t>(sizeof(bool) * host_n_rows[v.i] * host_n_rows[v.i], align);
      }
      RAFT_CUDA_TRY(cudaMemcpyAsync(adj_group_offset,
                                    h_adj_group_offset,
                                    this->n_groups * sizeof(std::size_t),
                                    cudaMemcpyHostToDevice,
                                    stream));
    }
    return;
  }

  void destroy()
  {
    adj_col_stride   = nullptr;
    adj_group_offset = nullptr;
    if (h_adj_group_offset == nullptr) {
      delete[] h_adj_group_offset;
      // cudaFreeHost(h_adj_group_offset);
      h_adj_group_offset = nullptr;
    }
    return;
  }

  bool optim_layout = false;
  const Index_t max_nbr;
  Data_t* adj;
  Index_t* adj_col_stride       = nullptr;  // stride of cols for different groups
  std::size_t* adj_group_offset = nullptr;  // offset of groups
  size_t wsp_size;

 private:
  std::size_t* h_adj_group_offset = nullptr;
};

template <typename Data_t,
          typename Index_t       = int,
          typename MetaDataClass = MultiGroupMetaData<Index_t>,
          typename BaseClass     = BaseAccessor<Data_t, Index_t, MetaDataClass>>
class CorePointAccessor : public BaseClass {
 public:
  CorePointAccessor(const MetaDataClass* _metadata, Data_t* _data)
    : BaseClass(_metadata, _data), core_pts(_data)
  {
  }

  size_t get_wsp_size()
  {
    return raft::alignTo<std::size_t>(this->n_points * sizeof(Index_t), align);
  }

  void initialize(const raft::handle_t& handle,
                  void* workspace,
                  size_t buffer_size,
                  cudaStream_t stream)
  {
    wsp_size = this->get_wsp_size();
    ASSERT(buffer_size == wsp_size,
           "The required size of workspace (%ld) doesn't match that passed (%ld) in %s.\n",
           wsp_size,
           buffer_size,
           __FUNCTION__);

    if (offset_mask != nullptr) { this->destroy(); }

    offset_mask = reinterpret_cast<Index_t*>(workspace);
    dim3 gridSize{static_cast<unsigned int>(this->n_groups)};
    dim3 blkSize{64};
    init_offset_mask<<<gridSize, blkSize, 0, stream>>>(
      offset_mask, this->n_rows_ptr, this->row_start_ids, this->n_groups);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    return;
  }

  void destroy()
  {
    offset_mask = nullptr;
    return;
  }

  Data_t* core_pts;
  Index_t* offset_mask = nullptr;
  size_t wsp_size;
};

}  // namespace Metadata
}  // namespace Multigroups
}  // namespace Dbscan
}  // namespace ML