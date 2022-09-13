#pragma once
#include <cstddef>
#include <new>

namespace herring {
namespace detail {
#ifdef __cpplib_hardware_interference_size
using std::hardware_constructive_interference_size;
#else
auto constexpr static const hardware_constructive_interference_size=std::size_t{64};
#endif
}
}
