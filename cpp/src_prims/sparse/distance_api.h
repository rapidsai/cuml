#include <cusparse_v2.h>
#include <cuml/common/cuml_allocator.hpp>

#pragma once

namespace MLCommon {
namespace Sparse {
namespace Distance {


template <typename value_idx, typename value_t>
struct distances_config_t {
  // left side
  value_idx a_nrows;
  value_idx a_ncols;
  value_idx a_nnz;
  value_idx *a_indptr;
  value_idx *a_indices;
  value_t *a_data;

  // right side
  value_idx b_nrows;
  value_idx b_ncols;
  value_idx b_nnz;
  value_idx *b_indptr;
  value_idx *b_indices;
  value_t *b_data;

  cusparseHandle_t handle;

  std::shared_ptr<deviceAllocator> allocator;
  cudaStream_t stream;
};

template <typename value_t>
class distances_t {
 public:
  virtual void compute(value_t *out) { CUML_LOG_DEBUG("INside base"); }
  virtual ~distances_t() = default;
};


};
}
};