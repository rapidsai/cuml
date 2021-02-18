#include <specializations/raft/mr/host/buffer.hpp>

namespace raft {
namespace mr {

template class buffer_base<int, raft::mr::host::allocator>;
template class buffer_base<unsigned int, raft::mr::host::allocator>;
template class buffer_base<float, raft::mr::host::allocator>;
template class buffer_base<double, raft::mr::host::allocator>;
template class buffer_base<char, raft::mr::host::allocator>;
template class buffer_base<bool, raft::mr::host::allocator>;
template class buffer_base<long, raft::mr::host::allocator>;

namespace host {

template class buffer<float>;
template class buffer<int>;
template class buffer<unsigned int>;

}  // namespace host
}  // namespace mr
}  // namespace raft
