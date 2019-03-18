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
}
