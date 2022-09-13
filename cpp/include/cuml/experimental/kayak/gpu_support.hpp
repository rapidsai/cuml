#pragma once
#include <exception>
#include <cstddef>
#include <stdint.h>

namespace kayak {
#ifdef ENABLE_GPU
auto constexpr static const GPU_ENABLED = true;
#else
auto constexpr static const GPU_ENABLED = false;
#endif

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
auto constexpr static const GPU_COMPILATION = true;
#else
#define HOST
#define DEVICE
auto constexpr static const GPU_COMPILATION = false;
#endif

#ifndef DEBUG
auto constexpr static const DEBUG_ENABLED = false;
#elif DEBUG == 0
auto constexpr static const DEBUG_ENABLED = false;
#else
auto constexpr static const DEBUG_ENABLED = true;
#endif

struct gpu_unsupported : std::exception {
  gpu_unsupported() : gpu_unsupported("GPU functionality invoked in non-GPU build") {}
  gpu_unsupported(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

}
