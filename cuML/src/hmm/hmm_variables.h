#pragma once

/** Train options for HMM */
enum TrainOption {
        Vitebri,
        ForwardBackward
};

template <typename T>
struct Multinomial {
        T *dPis, *dLlhd;
        int lddPis, lddLlhd;

        int nCl, nDim, nObs;
};

template <typename T>
struct GMM {
        T *dmu, *dsigma, *dPis, *dPis_inv, *dLlhd;
        T **dX_array=NULL, **dmu_array=NULL, **dsigma_array=NULL;

        int lddx, lddmu, lddsigma, lddsigma_full, lddPis, lddLlhd;

        int nCl, nDim, nObs, nSeq;
};

template <typename T>
struct HMM {
        int nStates;

        // Transition and emission matrixes
        T *dT, *dB;
        // All dLlhd point to dGamma
        T *dAlpha, *dBeta, *dGamma;
        int lddt, lddb, lddalpha, lddbeta;

        T **dAlpha_array, **dBeta_array, **dGamma_array;

        int** dMaxPath_array;

        T* alphaScaleCoefs;

        int **dV_idx_array;
        T **dV_array;

        TrainOption train;
};

template <typename T>
struct GMMHMM {
        std::vector<GMM<T> > gmms;
        HMM hmm;
};

template <typename T>
struct MultinomialHMM {
        std::vector<Multinomial<T> > multinomials;
        HMM hmm;
};

}
