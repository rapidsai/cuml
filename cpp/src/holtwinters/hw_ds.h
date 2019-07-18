#pragma once

#pragma once

#define MAX_BLOCKS_PER_DIM 65535

#define GET_TID (blockIdx.x * blockDim.x + threadIdx.x)

inline int GET_THREADS_PER_BLOCK(const int n, const int max_threads = 512) {
  int ret;
  if (n <= 128)
    ret = 32;
  else if (n <= 1024)
    ret = 128;
  else
    ret = 512;
  return ret > max_threads ? max_threads : ret;
}

inline int GET_NUM_BLOCKS(const int n, const int max_threads = 512,
                          const int max_blocks = MAX_BLOCKS_PER_DIM) {
  int ret = (n - 1) / GET_THREADS_PER_BLOCK(n, max_threads) + 1;
  return ret > max_blocks ? max_blocks : ret;
}

namespace ML {

enum SeasonalType { ADDITIVE, MULTIPLICATIVE };

enum OptimCriterion {
  OPTIM_BFGS_ITER_LIMIT = 0,
  OPTIM_MIN_PARAM_DIFF = 1,
  OPTIM_MIN_ERROR_DIFF = 2,
  OPTIM_MIN_GRAD_NORM = 3,
};

template <typename Dtype>
struct OptimParams {
  Dtype eps;
  Dtype min_param_diff;
  Dtype min_error_diff;
  Dtype min_grad_norm;
  int bfgs_iter_limit;
  int linesearch_iter_limit;
  Dtype linesearch_tau;
  Dtype linesearch_c;
  Dtype linesearch_step_size;
};

enum Norm { L0, L1, L2, LINF };

}  // namespace ML