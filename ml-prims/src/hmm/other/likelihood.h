#pragma once

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
struct GMMLikelihood
{
        T *data, *mus, *sigmas;
        T* ps;
        int nCl, nDim, nObs;

        bool isLog;

        // cusolverDnHandle_t cusolverHandle;
        cublasHandle_t *cublasHandle;
        Determinant<T> *Det;
        Inverse<T> *Inv;

        T* temp, *inv_sigma, *temp_rhos;
        T epsilon = pow(10, -6);

        GMMLikelihood (T *data, T *mus, T *sigmas, T *ps, int _nCl,
                       int _nDim, int _nObs, bool _isLog, cublasHandle_t *_cublasHandle,
                       cusolverDnHandle_t *_cusolverHandle){
                this->data = data;
                this->mus = mus;
                this->sigmas = sigmas;
                this->ps = ps;

                // cusolverHandle = _cusolverHandle;
                cublasHandle = _cublasHandle;

                nCl = _nCl;
                nDim = _nDim;
                nObs = _nObs;

                isLog = _isLog;

                this->Det = new Determinant<T>(nDim, _cusolverHandle);
                this->Inv = new Inverse<T>(nDim, _cusolverHandle);

                // Temporary memory allocations
                allocate(temp, nDim);
                allocate(inv_sigma, nDim * nDim);
                temp_rhos = (T *)malloc(nCl * nObs * sizeof(T));
        }

        T _sample_gaussian_lhd(T* x, T* mu, T* sigma){
                // Computes log likelihood for normal distribution
                T logl=0;

                // Compute the squared sum
                Inv->compute(sigma, inv_sigma);

                // x - mu
                subtract(temp, x, mu, nDim);

                // sigma * (x - mu)
                bilinear(inv_sigma, nDim, temp, *cublasHandle, &logl);
                logl *= -0.5;

                T det = Det->compute(sigma);
                logl += -0.5 * std::log(det);

                logl += -0.5 * nDim * std::log(2 * M_PI);
                return logl;
        }

        T _class_sample_llhd(int sampleId, int classId)
        {

                T rho_h;
                updateHost(&rho_h, ps + classId, 1);
                return rho_h * _sample_gaussian_lhd(data + nDim * sampleId,
                                                    mus + nDim * classId,
                                                    sigmas + nDim * nDim * classId);
        }

        T set_llhd(){
                T logl = 0;

                for (int c_id = 0; c_id < nCl; c_id++) {
                        for (int s_id = 0; s_id < nObs; s_id++) {
                                logl += _class_sample_llhd(s_id, c_id);
                        }
                }

                if (!isLog)
                        return exp(logl);
                return logl;
        }

        void fill_rhos(T* rhos, T* ps)
        {
                T llhd;
                for (int classId = 0; classId < nCl; classId++) {
                        for (int sampleId = 0; sampleId < nObs; sampleId++) {
                                llhd = _sample_gaussian_lhd(data + nDim * sampleId,
                                                            mus + nDim * classId,
                                                            sigmas + nDim * nDim * classId);
                                if(!isLog) {
                                        llhd = max(exp(llhd), epsilon);
                                }
                                temp_rhos[IDX2C(sampleId, classId, nObs)] = llhd;
                        }
                }
                updateDevice(rhos, temp_rhos, nCl * nObs);
                print_matrix(rhos, nObs, nCl, "rhos_lhd");
                // print_matrix(ps, 1, nCl, "ps");
                CUBLAS_CHECK(cublasdgmm(*cublasHandle, CUBLAS_SIDE_RIGHT, nObs,
                                        nCl, rhos, nObs, ps, 1, rhos, nObs));
                print_matrix(rhos, nObs, nCl, "rhos_ after");

        }

        void TearDown() {
                Inv->TearDown();
                Det->TearDown();

                free(temp_rhos);
                CUDA_CHECK(cudaFree(temp));

        }
};



}
}
