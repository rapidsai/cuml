#pragma once

#include <hmm/magma/b_likelihood.h>

namespace multinomial {
void update_llhd(T* dX, Multinomial<T>& multinomial){
        for (size_t stateId = 0; stateId < hmm.nStates; stateId++) {
                MLCommon::multinomial_likelihood_batched(hmm.nObs,
                                                         hmm.nCl,
                                                         hmm.nStates,
                                                         dX,
                                                         hmm.dPb_array[stateId],
                                                         hmm.dists[stateId].dLlhd,
                                                         hmm.dists[stateId].lddLlhd,
                                                         false);
        }
}
}
