#include <raft/core/comms.hpp>
// #include <raft/core/handle.hpp>
// #include <raft/core/device_mdarray.hpp>
// #include <raft/util/cudart_utils.hpp>
// #include <raft/comms/std_comms.hpp>
#include <cuml/common/logger.hpp>
#include <cuml/linear_model/qn.h> // to use qn_params

#include <cuda_runtime.h>

// #include <vector>
#include <iostream>

namespace ML {
namespace GLM {
namespace opg {

void toy(const raft::handle_t &handle, 
           float* X,
           int N,
           int D) ;

void qnFit(const raft::handle_t &handle, 
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

/**
 * @brief Fit a GLM using quasi newton methods.
 *
 * @param cuml_handle   reference to raft::handle_t object
 * @param params        model parameters
 * @param X             device pointer to a contiguous feature matrix of dimension [N, D]
 * @param X_col_major   true if X is stored column-major
 * @param y             device pointer to label vector of length N
 * @param N             number of examples
 * @param D             number of features
 * @param C             number of outputs (number of classes or `1` for regression)
 * @param w0            device pointer of size (D + (fit_intercept ? 1 : 0)) * C with initial point,
 *                      overwritten by final result.
 * @param f             host pointer holding the final objective value
 * @param num_iters     host pointer holding the actual number of iterations taken
 * @param sample_weight device pointer to sample weight vector of length n_rows (nullptr
   for uniform weights)
 * @param svr_eps       epsilon parameter for svr
 */
/*
template <typename T, typename I = int>
void qnFit(const raft::handle_t& cuml_handle,
           const qn_params& params,
           float* X,
           bool X_col_major,
           float* y,
           int N,
           int D,
           int C,
           float* w0,
           float* f,
           int* num_iters,
           float* sample_weight = nullptr,
           T svr_eps        = 0);
           */

};  // namespace opg
};  // namespace GLM
};  // namespace ML



