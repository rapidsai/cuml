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
struct GMMLikelihood
{
        T *data, *mus, *sigmas;
        T* rhos;
        int nCl, nDim, nObs;

        bool isLog;

        cublasHandle_t *handle;
        Determinant<T> *Det;
        Inverse<T> *Inv;

        GMMLikelihood (T *data, T *mus, T *sigmas, T *rhos, int _nCl,
                       int _nDim, int _nObs, bool _isLog, cublasHandle_t *handle
                       ){
                this->data = data;
                this->mus = mus;
                this->sigmas = sigmas;
                this->rhos = rhos;

                this->handle = handle;

                nCl = _nCl;
                nDim = _nDim;
                nObs = _nObs;

                isLog = _isLog;

                this->Det = new Determinant<T>(nDim);
                this->Inv = new Inverse<T>(nDim);
        }

        T _sample_gaussian_lhd(int sampleId, int classId){
                // Computes log likelihood for normal distribution
                T logl=0, det =0;

                T* inv_sigma, *temp;

                allocate(inv_sigma, nDim);
                allocate(temp, nDim);

                Inv->compute(sigmas + nDim * nDim * classId, inv_sigma);

                // x - mu
                subtract(temp, data + nDim * sampleId, mus + nDim * classId, nDim);

                // Compute the squared sum
                bilinear(inv_sigma, nDim, temp, *handle, &logl);
                logl *= -0.5;

                det = Det->compute(sigmas + nDim * nDim * classId);
                logl += -0.5 * std::log(det);

                logl += -0.5 * nDim * std::log(2 * M_PI);
                return logl;
        }

        T _class_sample_llhd(int sampleId, int classId)
        {
                T rho_h;
                updateHost(&rho_h, rhos + classId, 1);
                return rho_h * _sample_gaussian_lhd(sampleId, classId);
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

        // void fill_rhos(int sampleId, int classId)
        // {
        //         *(rhos + classId + sampleId * nObs) = std::exp(_sample_gaussian_lhd(data + nDim * sampleId,
        //                                                                             mus + nDim * classId,
        //                                                                             sigmas + nDim * nDim * classId,
        //                                                                             nDim, *handle, *Det));
        // }
};



}
}
