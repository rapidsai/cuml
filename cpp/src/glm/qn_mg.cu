#include <raft/core/comms.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/core/handle.hpp>

#include <cuml/common/logger.hpp>

//#include "glm/qn/glm_base.cuh"
//#include "glm/qn/glm_logistic.cuh"
//#include "glm/qn/qn_util.cuh"
//#include "glm/qn/qn_solvers.cuh"
//#include "glm/qn/glm_regularizer.cuh"

#include <cuml/linear_model/qn.h> // to use qn_params
#include "qn/simple_mat/dense.hpp"
#include "qn/qn_util.cuh"
#include "qn/glm_logistic.cuh"
#include "qn/glm_regularizer.cuh"

#include "qn/glm_base_mg.cuh"

#include <cuda_runtime.h>
#include <iostream>


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
                const qn_params& pams,
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

  auto X_simple = SimpleDenseMat<T>(X, N, D, X_col_major? COL_MAJOR : ROW_MAJOR);
  std::cout << raft::arr2Str(X_simple.data, N * D, "X data", stream).c_str() << std::endl;

  auto y_simple = SimpleVec<T>(y, N);
  std::cout << raft::arr2Str(y_simple.data, N * 1, "y_sample data", stream).c_str() << std::endl;

  SimpleVec<T> coef_simple(w0, D + pams.fit_intercept);

  std::cout << "DEBUG: before coef_simple arr2Str" << std::endl;
  std::cout << "rank " << rank << ", D: " << D << ", fit_intercept: " << pams.fit_intercept << ", C: " << C << std::endl;

  auto coef_size = (D + pams.fit_intercept) * (C == 2 ? 1 : C); 
  std::cout << "rank " << rank << ", coef_size: " << coef_size << std::endl;
  std::cout << raft::arr2Str(coef_simple.data, 1 , "coef data[0]: ", stream).c_str() << std::endl;
  std::cout << raft::arr2Str(coef_simple.data, coef_size, "coef data", stream).c_str() << std::endl;
  std::cout << "DEBUG: after coef_simple arr2Str" << std::endl;

  // prepare configs opt_param
  //qn_params pams;
  //pams.loss = QN_LOSS_LOGISTIC;
  //pams.penalty_l2 = 1;
  //pams.change_tol = 1e-6;
  ML::GLM::detail::LBFGSParam<T> opt_param(pams);


  // prepare regularizer regularizer_obj
  ML::GLM::detail::LogisticLoss<T> loss_func(handle, D, pams.fit_intercept);
  T l1 = pams.penalty_l1;
  T l2 = pams.penalty_l2;
  if (pams.penalty_normalized) {
      l1 /= n_samples; // l1 /= 1/X.m
      l2 /= n_samples; // l2 /= 1/X.m
  }
  ML::GLM::detail::Tikhonov<T> reg(l2);
  ML::GLM::detail::RegularizedGLM<T, ML::GLM::detail::LogisticLoss<T>, decltype(reg)> regularizer_obj(&loss_func, &reg);


  // prepare GLMWithDataMG
  int n_targets = C == 2 ? 1 : C;
  rmm::device_uvector<T> tmp(n_targets * N, stream);
  SimpleDenseMat<T> Z(tmp.data(), n_targets, N);
  auto obj_function = GLMWithDataMG(handle, rank, n_ranks, n_samples, &regularizer_obj, X_simple, y_simple, Z);

  // prepare temporary variables fx, k, workspace 
  float fx = -1; 
  int k = -1;
  rmm::device_uvector<float> tmp_workspace(lbfgs_workspace_size(opt_param, coef_simple.len), stream);
  SimpleVec<float> workspace(tmp_workspace.data(), tmp_workspace.size());

  // call min_lbfgs
  min_lbfgs(opt_param, obj_function, coef_simple, fx, &k, workspace, stream, 5);
  std::cout << raft::arr2Str(coef_simple.data, 3, "coef result", stream).c_str() << std::endl;

}

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
           int n_ranks)
{
  qnFit_impl<float, int>(
    handle,
    pams,
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
