#include <raft/core/comms.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/core/handle.hpp>
#include <raft/linalg/multiply.cuh>

#include "glm/qn/glm_base.cuh"
#include "glm/qn/glm_logistic.cuh"
#include "glm/qn/qn_util.cuh"
#include "glm/qn/qn_solvers.cuh"
#include "glm/qn/glm_regularizer.cuh"
#include <vector>
#include <iostream>

namespace ML {
namespace GLM {
namespace opg {
template<typename T>
// multi-gpu version of linearBwd
inline void linearBwdMG(const raft::handle_t& handle,
                      SimpleDenseMat<T>& G,
                      const SimpleMat<T>& X,
                      const SimpleDenseMat<T>& dZ,
                      bool setZero,
                      const int64_t n_samples, 
                      const int n_ranks)
{
  cudaStream_t stream = handle.get_stream();
  // Backward pass:
  // - compute G <- dZ * X.T
  // - for bias: Gb = mean(dZ, 1)

  const bool has_bias = X.n != G.n;
  const int D         = X.n;
  const T beta        = setZero ? T(0) : T(1);

  if (has_bias) {

    SimpleVec<T> Gbias;
    SimpleDenseMat<T> Gweights;

    col_ref(G, Gbias, D);

    col_slice(G, Gweights, 0, D);

    // TODO can this be fused somehow?
    const SimpleDenseMat<T>* X_simple_p = (const SimpleDenseMat<T>*)(&X);
    Gweights.assign_gemm(handle, 1.0 / n_samples, dZ, false, X, false, beta / n_ranks, stream);

    raft::stats::mean(Gbias.data, dZ.data, dZ.m, dZ.n, false, true, stream);
    T bias_factor = 1.0 * dZ.n / n_samples;
    raft::linalg::multiplyScalar(Gbias.data, Gbias.data, bias_factor, dZ.m, stream);

  } else {
    CUML_LOG_DEBUG("has bias not enabled");
    G.assign_gemm(handle, 1.0 / n_samples, dZ, false, X, false, beta / n_ranks, stream);
  }
}

template <typename T, class GLMObjective>
struct GLMWithDataMG : ML::GLM::detail::GLMWithData<T, GLMObjective> {

    const raft::handle_t* handle_p;
    int rank;
    int64_t n_samples;
    int n_ranks;

    GLMWithDataMG(raft::handle_t const &handle, int rank, int n_ranks, int64_t n_samples, GLMObjective* obj, const SimpleMat<T>& X, const SimpleVec<T>& y, SimpleDenseMat<T>& Z)
        : ML::GLM::detail::GLMWithData<T, GLMObjective> (obj, X, y, Z)
    {
        this->handle_p = &handle;
        this->rank = rank;
        this->n_ranks = n_ranks;
        this->n_samples = n_samples;
    }

    inline T operator()(const SimpleVec<T>& wFlat,
                        SimpleVec<T>& gradFlat,
                        T* dev_scalar,
                        cudaStream_t stream)
    {
        SimpleDenseMat<T> W(wFlat.data, this->C, this->dims);
        SimpleDenseMat<T> G(gradFlat.data, this->C, this->dims);
        SimpleVec<T> lossVal(dev_scalar, 1);

        // apply regularization
        auto regularizer_obj = this->objective;
        auto lossFunc = regularizer_obj->loss;
        auto reg = regularizer_obj->reg;
        G.fill(0, stream);
        reg->reg_grad(dev_scalar, G, W, lossFunc->fit_intercept, stream);
        float reg_host;
        raft::update_host(&reg_host, dev_scalar, 1, stream);

        // apply linearFwd, getLossAndDz, linearBwd
        ML::GLM::detail::linearFwd(lossFunc->handle, *(this->Z), *(this->X), W);                  // linear part: forward pass

        raft::comms::comms_t const& communicator = raft::resource::get_comms(*(this->handle_p));   

        lossFunc->getLossAndDZ(dev_scalar, *(this->Z), *(this->y), stream);  // loss specific part

        // add normalization to distributed losses
        float tmp_host = -1.0;
        raft::update_host(&tmp_host, dev_scalar, 1, stream);
        tmp_host = tmp_host * (*this->y).len / this->n_samples;
        lossVal.fill(tmp_host, stream);
        raft::interruptible::synchronize(stream);

        communicator.allreduce(dev_scalar, dev_scalar, 1,                          
            raft::comms::op_t::SUM, stream);                                                                 
        raft::resource::sync_stream(*(this->handle_p));    
        raft::interruptible::synchronize(stream);

        linearBwdMG(lossFunc->handle, G, *(this->X), *(this->Z), false, n_samples, n_ranks);    // linear part: backward pass
        raft::interruptible::synchronize(stream);

        communicator.allreduce(G.data, G.data, this->C * this->dims,                          
            raft::comms::op_t::SUM, stream);                                                                 
        raft::resource::sync_stream(*(this->handle_p));    

        float loss_host;
        raft::update_host(&loss_host, dev_scalar, 1, stream);

        raft::interruptible::synchronize(stream);


        lossVal.fill(loss_host + reg_host, stream);
        raft::update_host(&loss_host, dev_scalar, 1, stream);
        raft::interruptible::synchronize(stream);

        return loss_host;
    }


};
};  // namespace OPG
};  // namespace GLM
};  // namespace ML