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
        T *dStartProb, *dT, *dB, *dAlpha, *dBeta, *dGamma;
        int lddsp, lddt, lddb, lddalpha, lddbeta, lddgamma;

        int nObs, nSeq, max_len;
        T* dLlhd;

        T **dB_array, **dPi_array;

        T *dV;
        int lddv;

        int nFeatures;
        unsigned short int *dcumlenghts_inc, *dcumlenghts_exc;

        TrainOption train;
        DistOption distOption;
};

}
