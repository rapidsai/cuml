#pragma once
#include <raft/mr/device/buffer.hpp>

namespace raft {
namespace mr {
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
