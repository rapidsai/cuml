#pragma once
#ifdef ENABLE_GPU
#include <cuda_runtime_api.h>
#endif

namespace kayak {
#ifdef ENABLE_GPU
using cuda_stream = cudaStream_t;
#else
using cuda_stream = int;
#endif
}
