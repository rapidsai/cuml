#pragma once

#include <stdlib.h>
#include <vector>


namespace gmm {
template <typename T>
struct GMM {
        T *dmu, *dsigma, *dPis, *dPis_inv, *dLlhd;
        T **dX_array=NULL, **dmu_array=NULL, **dsigma_array=NULL;

        int lddx, lddmu, lddsigma, lddsigma_full, lddPis, lddLlhd;

        int nCl, nDim, nObs;

        T reg_covar, *cur_llhd;

        T* dProbNorm;
        int lddprobnorm;
};

}

namespace hmm {

/** Train options for HMM */
enum TrainOption {
        Vitebri,
        ForwardBackward
};


template <typename T>
struct HMM {
        int nStates;
        std::vector<gmm::GMM<T> > gmms;

        // All dLlhd point to dGamma
        T *dT, *dB, *dAlpha, *dBeta, *dGamma;
        int lddt, lddb, lddalpha, lddbeta, lddgamma;

        T **dAlpha_array, **dBeta_array, **dGamma_array;

        int** dMaxPath_array;

        T* alphaScaleCoefs;

        int **dV_idx_array;
        T **dV_array;
        T **dT_pows;

        TrainOption train;
};

// template <typename T>
// struct GMMHMM {
//         //
//         HMM *hmm;
// };

// template <typename T>
// struct Multinomial {
//         T *dPis, *dLlhd;
//         int lddPis, lddLlhd;
//
//         int nCl, nDim, nObs;
// };
//
// template <typename T>
// struct MultinomialHMM {
//         std::vector<Multinomial<T> > multinomials;
//         HMM hmm;
// };

}
