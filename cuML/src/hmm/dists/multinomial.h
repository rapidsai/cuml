#pragma once

#include <hmm/magma/b_likelihood.h>
#include <hmm/dists/dists_variables.h>
#include <hmm/hmm_variables.h>


namespace multinomial {

template <typename T>
void update_llhd(int* dX, hmm::HMM<T, Multinomial<T> >& hmm){
        // Create dPi array
        // MLCommon::multinomial_likelihood_batched(hmm.nObs,
        //                                          hmm.nCl,
        //                                          hmm.nStates,
        //                                          dX,
        //                                          hmm.dPi_array[stateId],
        //                                          hmm.dists[stateId].dLlhd,
        //                                          hmm.dists[stateId].lddLlhd,
        //                                          false);
}

template <typename T>
void init_multinomial(Multinomial<T>& multinomial,
                      T* dPis, int nCl) {
        multinomial.dPis = dPis;
        multinomial.nCl = nCl;

}
}
