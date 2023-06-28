#include "glm_logistic.cuh"
#include "glm_softmax.cuh"
#include "glm_regularizer.cuh"
#include "glm_softmax.cuh"
#include "glm_svm.cuh"
#include "qn_solvers.cuh"
#include "qn_util.cuh"
#include "glm_base_mg.cuh"

#include <cuml/linear_model/qn.h>

#include <raft/matrix/math.cuh>
#include <rmm/device_uvector.hpp>
namespace ML {
namespace GLM {
namespace opg {
using namespace ML::GLM::detail;

template <typename T, typename LossFunction>
int qn_fit_mg(const raft::handle_t& handle,
           const qn_params& pams,
           LossFunction& loss,
           const SimpleMat<T>& X,
           const SimpleVec<T>& y,
           SimpleDenseMat<T>& Z,
           T* w0_data,  // initial value and result
           T* fx,
           int* num_iters,
           int64_t n_samples,
           int rank,
           int n_ranks)
{
  cudaStream_t stream = handle.get_stream();
  LBFGSParam<T> opt_param(pams);
  SimpleVec<T> w0(w0_data, loss.n_param);

  // Scale the regularization strength with the number of samples.
  T l2 = pams.penalty_l2;
  if (pams.penalty_normalized) {
    l2 /= n_samples;
  }

  if (l2 == 0) {
    GLMWithDataMG<T, LossFunction> lossWith(handle, rank, n_ranks, n_samples, &loss, X, y, Z);

    return ML::GLM::detail::qn_minimize(handle, w0, fx, num_iters, lossWith, l1, opt_param, pams.verbose);

  } else {
    ML::GLM::detail::Tikhonov<T> reg(l2);
    ML::GLM::detail::RegularizedGLM<T, LossFunction, decltype(reg)> obj(&loss, &reg);
    GLMWithDataMG<T, decltype(obj)> lossWith(handle, rank, n_ranks, n_samples, &obj, X, y, Z);

    return ML::GLM::detail::qn_minimize(handle, w0, fx, num_iters, lossWith, l1, opt_param, pams.verbose);
  }
}

template <typename T>
inline void qn_fit_x_mg(const raft::handle_t& handle,
                     const qn_params& pams,
                     SimpleMat<T>& X,
                     T* y_data,
                     int C,
                     T* w0_data,
                     T* f,
                     int* num_iters,
                     int64_t n_samples,
                     int rank,
                     int n_ranks,
                     T* sample_weight = nullptr,
                     T svr_eps        = 0)
{
  /*
   NB:
    N - number of data rows
    D - number of data columns (features)
    C - number of output classes

    X in R^[N, D]
    w in R^[D, C]
    y in {0, 1}^[N, C] or {cat}^N

    Dimensionality of w0 depends on loss, so we initialize it later.
   */
  cudaStream_t stream = handle.get_stream();
  int N               = X.m;
  int D               = X.n;
  int n_targets       = ML::GLM::detail::qn_is_classification(pams.loss) && C == 2 ? 1 : C;
  rmm::device_uvector<T> tmp(n_targets * N, stream);
  SimpleDenseMat<T> Z(tmp.data(), n_targets, N);
  SimpleVec<T> y(y_data, N);

  switch (pams.loss) {
    case QN_LOSS_LOGISTIC: {
      ASSERT(C == 2, "qn.h: logistic loss invalid C");
      ML::GLM::detail::LogisticLoss<T> loss(handle, D, pams.fit_intercept);
      ML::GLM::opg::qn_fit_mg<T, decltype(loss)>(handle, pams, loss, X, y, Z, w0_data, f, num_iters, n_samples, rank, n_ranks);
    } break;
    case QN_LOSS_SOFTMAX: {
      ASSERT(C > 2, "qn.h: softmax invalid C");
      ML::GLM::detail::Softmax<T> loss(handle, D, C, pams.fit_intercept);
      ML::GLM::opg::qn_fit_mg<T, decltype(loss)>(handle, pams, loss, X, y, Z, w0_data, f, num_iters, n_samples, rank, n_ranks);
    } break;
    default: {
      ASSERT(false, "qn.h: unknown loss function type (id = %d).", pams.loss);
    }
  }
}

};  // namespace opg 
};  // namespace GLM
};  // namespace ML