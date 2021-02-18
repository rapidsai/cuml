#include <specializations/raft/mr/device/buffer.hpp>

namespace raft {
namespace mr {

template class buffer_base<int, raft::mr::device::allocator>;
template class buffer_base<int*, raft::mr::device::allocator>;
template class buffer_base<unsigned int, raft::mr::device::allocator>;
template class buffer_base<float, raft::mr::device::allocator>;
template class buffer_base<float*, raft::mr::device::allocator>;
template class buffer_base<double, raft::mr::device::allocator>;
template class buffer_base<double*, raft::mr::device::allocator>;
template class buffer_base<char, raft::mr::device::allocator>;
template class buffer_base<long, raft::mr::device::allocator>;
template class buffer_base<unsigned long, raft::mr::device::allocator>;
template class buffer_base<unsigned long long, raft::mr::device::allocator>;
template class buffer_base<bool, raft::mr::device::allocator>;

namespace device {

template class buffer<float>;
template class buffer<int>;
template class buffer<unsigned int>;
template class buffer<double>;
template class buffer<char>;
template class buffer<long>;

}  // namespace device
}  // namespace mr
}  // namespace raft
