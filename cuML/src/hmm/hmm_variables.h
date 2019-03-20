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

enum DistOption {
        MultinomialDist,
        GaussianMixture
};


// D is the emission distribution
template <typename T, typename D>
struct HMM {
        int nStates;
        std::vector<D> dists;
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
        DistOption distOption;
};

}
