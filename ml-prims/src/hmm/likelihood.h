#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <random/rng.h>

#include <linalg/cublas_wrappers.h>
#include <linalg/subtract.h>
#include "linalg/eltwise.h"
#include "linalg/transpose.h"

#include "hmm/bilinear.h"

#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>

#include "hmm/utils.h"
#include "hmm/determinant.h"
#include "hmm/inverse.h"


using namespace MLCommon::LinAlg;
using MLCommon::LinAlg::scalarMultiply;

namespace MLCommon {
namespace HMM {

template <typename T>
// __host__ __device__
T _sample_gaussian_lhd(T* x, T* mu, T* sigma, int nDim, cublasHandle_t handle,
                       Determinant<T> Det, Inverse<T> Inv){
        // Computes log likelihood for normal distribution
        T logl=0;

        // Compute the squared sum
        T* temp;
        allocate(temp, nDim);

        T* inv_sigma;
        allocate(inv_sigma, nDim);
        Inv.compute(sigma, inv_sigma);

        // x - mu
        subtract(temp, x, mu, nDim);

        // sigma * (x - mu)
        bilinear(inv_sigma, nDim, temp, handle, &logl);
        logl *= -0.5;

        T det = Det.compute(sigma);
        logl += -0.5 * std::log(det);

        logl += -0.5 * nDim * std::log(2 * M_PI);
        return logl;
}

template <typename T>
// __host__ __device__
struct _gmm_likelihood_functor
{
        T *data, *mus, *sigmas;
        T* rhos;
        int nCl, nDim, nObs;
        cublasHandle_t *handle;
        Determinant<T> *Det;
        Inverse<T> *Inv;

        _gmm_likelihood_functor (T *data, T *mus, T *sigmas, T *rhos, int _nCl,
                                 int _nDim, int _nObs, cublasHandle_t *handle,
                                 Determinant<T> *_Det, Inverse<T> *_Inv){
                this->data = data;
                this->mus = mus;
                this->sigmas = sigmas;
                this->rhos = rhos;

                this->handle = handle;
                this->Det = _Det;
                this->Inv = _Inv;

                nCl = _nCl;
                nDim = _nDim;
                nObs = _nObs;
        }

        // __host__ __device__
        // T operator()(int sampleId, int classId)
        T compute_llhd(int sampleId, int classId)
        {

                T rho_h;
                updateHost(&rho_h, rhos + classId, 1);
                T g =   _sample_gaussian_lhd(data + nDim * sampleId,
                                             mus + nDim * classId,
                                             sigmas + nDim * nDim * classId,
                                             nDim, *handle, *Det, *Inv);

                return (T) rho_h * g;
        }
        // __host__ __device__
        // void fill_rhos(int sampleId, int classId)
        // {
        //         *(rhos + classId + sampleId * nObs) = std::exp(_sample_gaussian_lhd(data + nDim * sampleId,
        //                                                                             mus + nDim * classId,
        //                                                                             sigmas + nDim * nDim * classId,
        //                                                                             nDim, *handle, *Det));
        // }
};

template <typename T>
T set_gmm_lhd(T* data, T* mus, T* sigmas, T* rhos, bool isLog,
              int nCl, int nDim, int nObs, cublasHandle_t *handle){
        Determinant<T> Det(nDim);
        Inverse<T> Inv(nDim);
        _gmm_likelihood_functor<T> gaussian_llhd_op(data, mus, sigmas, rhos,
                                                    nCl, nDim, nObs, handle,
                                                    &Det, &Inv);
        T logl = 0;

        for (int c_id = 0; c_id < nCl; c_id++) {
                for (int s_id = 0; s_id < nObs; s_id++) {
                        logl += gaussian_llhd_op.compute_llhd(s_id, c_id);
                }
        }

        if (!isLog)
                return exp(logl);
        return logl;
}

}
}
