#pragma once

template <typename T>
struct GMM {
        T *dmu, *dsigma, *dPis, *dPis_inv, *dLlhd;
        T **dX_array=NULL, **dmu_array=NULL, **dsigma_array=NULL;

        int lddx, lddmu, lddsigma, lddsigma_full, lddPis, lddLlhd;

        int nCl, nDim, nObs;
};

template <typename T>
struct HMM {
        GMM<T>& gmm;

        T *dT;
        int *dV, **dV_array;
        T **dAlpha_array, **dBeta_array
        int lddt, lddv;
};
