#include <random/rng.h>
#include <linalg/cublas_wrappers.h>
#include <linalg/subtract.h>
#include "linalg/eltwise.h"
#include "linalg/transpose.h"
#include "hmm/bilinear.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>

#include "hmm/utils.h"

using namespace MLCommon::LinAlg;
using MLCommon::LinAlg::scalarMultiply;

namespace MLCommon {
namespace HMM {

template <typename T>
__host__ __device__
T _sample_gaussian_lhd(T* x, T* mu, T* sigma, int nDim, cublasHandle_t handle){
        // Computes log likelihood for normal distribution
        T logl = 0;

        // Compute the squared sum
        T* temp;
        allocate(temp, nDim);
        printf("line %d file %s\n", __LINE__, __FILE__);

        // x - mu
        subtract(temp, x, mu, nDim);
        scalarMultiply(temp, temp, (T) -0.5, nDim);
        printf("line %d file %s\n", __LINE__, __FILE__);

        // sigma * (x - mu)
        bilinear(sigma, nDim, temp, handle, &logl);
        printf("line %d file %s\n", __LINE__, __FILE__);

        // logl += 0.5 * std::log(2 * std::pi);
        T determinant = 0.;
        logl += determinant;
        return logl;
}

template <typename T>
__host__ __device__
struct _gmm_likelihood_functor
{
        T *data, *mus, *sigmas;
        T* rhos;
        int nCl, nDim, nObs;
        cublasHandle_t *handle;

        _gmm_likelihood_functor (T *data, T *mus, T *sigmas, T *rhos, int _nCl,
                                 int _nDim, int _nObs, cublasHandle_t *handle){
                this->data = data;
                this->mus = mus;
                this->sigmas = sigmas;
                this->rhos = rhos;

                this->handle = handle;

                nCl = _nCl;
                nDim = _nDim;
                nObs = _nObs;
        }

        __host__ __device__
        // T operator()(int sampleId, int classId)
        T compute_llhd(int sampleId, int classId)
        {
                printf("line %d file %s\n", __LINE__, __FILE__);
//  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                // TODO ::: Multiply by rhos[classId]
                return (T) _sample_gaussian_lhd(data + nDim * sampleId,
                                                mus + nDim * classId,
                                                sigmas + nDim * nDim * classId,
                                                nDim, *handle);
        }

        void fill_rhos(int sampleId, int classId)
        {
                rhos[classId + sampleId * nObs] = exp(compute_llhd(sampleId, classId));
        }
};

template <typename T>
T set_gmm_lhd(T* data, T* mus, T* sigmas, T* rhos, bool isLog,
              int nCl, int nDim, int nObs, cublasHandle_t *handle){
        T logl = 0;
        printf("line %d file %s\n", __LINE__, __FILE__);

        _gmm_likelihood_functor<T> gaussian_llhd_op(data, mus, sigmas, rhos,
                                                    nCl, nDim, nObs, handle);
        printf("line %d file %s\n", __LINE__, __FILE__);
        print_matrix(rhos, nCl, 1, "rhos");
        print_matrix(data, nDim, nObs, "data");
        print_matrix(mus, nDim, nCl, "mus");
        print_matrix(sigmas, nDim * nDim, nCl, "sigmas");
        for (int c_id = 0; c_id < nCl; c_id++) {
                for (int s_id = 0; s_id < nObs; s_id++) {
                        printf("(c, s) %d %d\n", c_id, s_id);
                        logl += gaussian_llhd_op.compute_llhd(s_id, c_id);
                }
        }

        if (!isLog)
                return exp(logl);
        return logl;
}

}
}
