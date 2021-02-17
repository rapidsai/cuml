#include <specializations/raft/mr/host/buffer.hpp>

namespace raft {
namespace mr {
namespace host {

template class buffer<float>;
template class buffer<int>;
template class buffer<unsigned int>;

}  // namespace host
}  // namespace mr
}  // namespace raft
