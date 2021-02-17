#pragma once
#include <raft/mr/host/buffer.hpp>

namespace raft {
namespace mr {
namespace host {
  extern template class buffer<float>;
  extern template class buffer<int>;
  extern template class buffer<unsigned int>;
}
}
}
