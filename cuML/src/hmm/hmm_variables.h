#pragma once

#include <stdlib.h>
#include <vector>

#include <gmm/gmm_variables.h>

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

        int nObs, nSeq, max_len;

        T **dAlpha_array, **dBeta_array, **dB_array;

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
