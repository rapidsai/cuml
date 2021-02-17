#include <specializations/raft/mr/device/buffer.hpp>

namespace raft {
namespace mr {
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
