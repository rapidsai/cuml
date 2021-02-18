#pragma once
#include <raft/mr/device/buffer.hpp>

namespace raft {
namespace mr {
  extern template class buffer_base<int, raft::mr::device::allocator>;
  extern template class buffer_base<int*, raft::mr::device::allocator>;
  extern template class buffer_base<unsigned int, raft::mr::device::allocator>;
  extern template class buffer_base<float, raft::mr::device::allocator>;
  extern template class buffer_base<float*, raft::mr::device::allocator>;
  extern template class buffer_base<double, raft::mr::device::allocator>;
  extern template class buffer_base<double*, raft::mr::device::allocator>;
  extern template class buffer_base<char, raft::mr::device::allocator>;
  extern template class buffer_base<long, raft::mr::device::allocator>;
  extern template class buffer_base<unsigned long, raft::mr::device::allocator>;
  extern template class buffer_base<unsigned long long, raft::mr::device::allocator>;
  extern template class buffer_base<bool, raft::mr::device::allocator>;
namespace device {
  extern template class buffer<float>;
  extern template class buffer<int>;
  extern template class buffer<unsigned int>;
  extern template class buffer<double>;
  extern template class buffer<char>;
  extern template class buffer<long>;
}
}
}
