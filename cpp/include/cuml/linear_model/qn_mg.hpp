#include <raft/core/comms.hpp>
#include <cuml/common/logger.hpp>
#include <cuml/linear_model/qn.h> 
#include <cuda_runtime.h>

namespace ML {
namespace GLM {
namespace opg {

void toy(const raft::handle_t &handle, 
           float* X,
           int N,
           int D) ;

void qnFit(const raft::handle_t &handle, 
           const qn_params& pams,
           float* X,
           bool X_col_major,
           float *y,
           int N,
           int D,
           int C,
           float* w0,
           float* f,
           int* num_iters,
           int n_samples,
           int rank,
           int n_ranks);

};  // namespace opg
};  // namespace GLM
};  // namespace ML



