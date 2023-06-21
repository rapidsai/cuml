#include <raft/core/comms.hpp>
// #include <raft/core/device_mdarray.hpp>
#include <raft/util/cudart_utils.hpp>
// #include <raft/comms/std_comms.hpp>

#include <cuml/common/logger.hpp>
#include <cuml/linear_model/qn.h> // to use qn_params

#include "glm/qn/glm_base.cuh"
#include "glm/qn/glm_logistic.cuh"
#include "glm/qn/qn_util.cuh"
#include "glm/qn/qn_solvers.cuh"
#include "glm/qn/glm_regularizer.cuh"

#include <cuda_runtime.h>
#include <iostream>

#include "glm/qn/glm_base.cuh"
#include "glm/qn/glm_logistic.cuh"
#include "glm/qn/qn_util.cuh"
#include "glm/qn/qn_solvers.cuh"
#include "glm/qn/glm_regularizer.cuh"

#include <cuml/linear_model/qn.h> // to use qn_params

namespace ML {
namespace GLM {
namespace opg {

void toy(const raft::handle_t &handle, 
           float* X,
           int N,
           int D) 
{
  std::cout << "hello world from qnFit" << std::endl;

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  // std::cout << raft::arr2Str(X, N * D, "X data", stream).c_str() << std::endl; 
}

template<typename T, typename I>
void qnFit_impl(const raft::handle_t &handle, 
                T* X,
                bool X_col_major,
                T *y,
                I N,
                I D,
                I C,
                T* w0,
                T* f,
                int* num_iters,
                I n_samples,
                int rank,
                int n_ranks) 
{
  std::cout << "hello world from qnFit" << std::endl;

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  std::cout << raft::arr2Str(X, N * D, "X data", stream).c_str() << std::endl;


  //std::vector<T> res = {1.};
  //raft::update_device(w0_data, res.data(), );
}

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
           int n_ranks)
{
  qnFit_impl<float, int>(
    handle,
    X,
    X_col_major,
    y,
    N,
    D,
    C,
    w0,
    f,
    num_iters,
    n_samples,
    rank,
    n_ranks);
}

};  // namespace OPG
};  // namespace GLM
};  // namespace ML

// #include <raft/core/device_mdarray.hpp>
//#include <iostream>
//#include <vector>

//#include <cuml/linear_model/qn_mg.hpp>

/*
namespace ML {
namespace GLM {


};  // namespace GLM
};  // namespace ML
*/

/*
template <typename T>
void qnFit(const raft::handle_t& handle,
           const qn_params& pams,
           T* X_data,
           bool X_col_major,
           T* y_data,
           int N,
           int D,
           int C,
           T* w0_data,
           T* f,
           int* num_iters,
           T* sample_weight = nullptr,
           T svr_eps        = 0)
{
}
*/
