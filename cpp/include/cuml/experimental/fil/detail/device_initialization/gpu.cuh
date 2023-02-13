#pragma once
#include <type_traits>
#include <cuda_runtime_api.h>
#include <cuml/experimental/fil/constants.hpp>
#include <cuml/experimental/fil/detail/gpu_introspection.hpp>
#include <cuml/experimental/fil/detail/infer_kernel/gpu.cuh>
#include <cuml/experimental/fil/detail/forest.hpp>
#include <cuml/experimental/fil/detail/specializations/device_initialization_macros.hpp>
#include <cuml/experimental/kayak/device_id.hpp>
#include <cuml/experimental/kayak/device_setter.hpp>
#include <cuml/experimental/kayak/device_type.hpp>
#include <cuml/experimental/kayak/gpu_support.hpp>
namespace ML {
namespace experimental {
namespace fil {
namespace detail {
namespace device_initialization {

/* The implementation of the template used to initialize GPU device options
 *
 * On GPU-enabled builds, the GPU specialization of this template ensures that
 * the inference kernels have access to the maximum available dynamic shared
 * memory.
 */
template<typename forest_t, kayak::device_type D>
std::enable_if_t<kayak::GPU_ENABLED && D==kayak::device_type::gpu, void> initialize_device(kayak::device_id<D> device) {
  auto device_context = kayak::device_setter(device);
  auto max_shared_mem_per_block = get_max_shared_mem_per_block(device);
  // Run solely for side-effect of caching SM count
  get_sm_count(device);
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<false, 1, forest_t, std::nullptr_t, std::nullptr_t>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<false, 2, forest_t>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<false, 4, forest_t>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<false, 8, forest_t>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<false, 16, forest_t>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<false, 32, forest_t>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<false, 1, forest_t, typename forest_t::io_type*>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<false, 2, forest_t, typename forest_t::io_type*>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<false, 4, forest_t, typename forest_t::io_type*>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<false, 8, forest_t, typename forest_t::io_type*>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<false, 16, forest_t, typename forest_t::io_type*>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<false, 32, forest_t, typename forest_t::io_type*>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 1, forest_t>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 2, forest_t>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 4, forest_t>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 8, forest_t>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 16, forest_t>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 32, forest_t>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 1, forest_t, typename forest_t::io_type*>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 2, forest_t, typename forest_t::io_type*>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 4, forest_t, typename forest_t::io_type*>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 8, forest_t, typename forest_t::io_type*>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 16, forest_t, typename forest_t::io_type*>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 32, forest_t, typename forest_t::io_type*>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 1, forest_t, std::nullptr_t, typename forest_t::node_type::index_type*>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 2, forest_t, std::nullptr_t, typename forest_t::node_type::index_type*>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 4, forest_t, std::nullptr_t, typename forest_t::node_type::index_type*>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 8, forest_t, std::nullptr_t, typename forest_t::node_type::index_type*>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 16, forest_t, std::nullptr_t, typename forest_t::node_type::index_type*>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 32, forest_t, std::nullptr_t, typename forest_t::node_type::index_type*>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 1, forest_t, typename forest_t::io_type*, std::nullptr_t>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 2, forest_t, typename forest_t::io_type*, std::nullptr_t>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 4, forest_t, typename forest_t::io_type*, std::nullptr_t>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 8, forest_t, typename forest_t::io_type*, std::nullptr_t>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 16, forest_t, typename forest_t::io_type*, std::nullptr_t>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 32, forest_t, typename forest_t::io_type*, std::nullptr_t>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 1, forest_t, typename forest_t::io_type*, typename forest_t::node_type::index_type*>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 2, forest_t, typename forest_t::io_type*, typename forest_t::node_type::index_type*>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 4, forest_t, typename forest_t::io_type*, typename forest_t::node_type::index_type*>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 8, forest_t, typename forest_t::io_type*, typename forest_t::node_type::index_type*>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 16, forest_t, typename forest_t::io_type*, typename forest_t::node_type::index_type*>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
  kayak::cuda_check(
    cudaFuncSetAttribute(
      infer_kernel<true, 32, forest_t, typename forest_t::io_type*, typename forest_t::node_type::index_type*>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_per_block
    )
  );
}

CUML_FIL_INITIALIZE_DEVICE(extern template, 0)
CUML_FIL_INITIALIZE_DEVICE(extern template, 1)
CUML_FIL_INITIALIZE_DEVICE(extern template, 2)
CUML_FIL_INITIALIZE_DEVICE(extern template, 3)
CUML_FIL_INITIALIZE_DEVICE(extern template, 4)
CUML_FIL_INITIALIZE_DEVICE(extern template, 5)
CUML_FIL_INITIALIZE_DEVICE(extern template, 6)
CUML_FIL_INITIALIZE_DEVICE(extern template, 7)

}
}
}

}
}
