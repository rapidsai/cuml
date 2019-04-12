/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <hmm/dists/dists_variables.h>
#include <gmm/gmm.h>
#include <hmm/hmm_variables.h>

namespace hmm {

template <typename T>
size_t get_workspace_size(gmm::GMM<T> &gmm){
        size_t workspaceSize = gmm_bufferSize(gmm);
        return workspaceSize;
}

template <typename T>
void create_handle(gmm::GMM<T> &gmm, void* workspace){
        create_GMMHandle_new(gmm, workspace);
// TODO : Assign the gmm dB to dProbNorm pointer for the specific hmm
// The emissions dB which is an attribute of the HMM will be updated implicitly when calling update_llhd for gmm, since the attribute dProbNorm will be pointing to the right pointer with the right offset of dB, which hold all the emissions
// It would look something like that :
// gmm.dProbNorm = hmm.dB + gmmId * nObs
// I am sure there is a cleaner way to implement it trougouht the whole framework
// I just thought of it, maybe the dB matrix would need to be tranposed
// Its leading dimension is over the number of states which is an issue
// Come to think of it, it may not be the best choice performance wise because I loop
// over the matrix observationwise in general for inference ...

}



// TODO : test this part
template <typename T>
void update_llhd(hmm::HMM<T, gmm::GMM<T> > &hmm,
                 T* dX,
                 cublasHandle_t cublasHandle,
                 magma_queue_t queue ){
        // for (size_t stateId = 0; stateId < hmm.nStates; stateId++) {
        //         update_llhd(dX, hmm.dists[stateId],
        //                     cublasHandle, queue);
        // }
}
//
//
// TODO : test this part
template <typename T>
void update_emissions(hmm::HMM<T, gmm::GMM<T> > &hmm,
                      T* dX){

// I should have rather called it fit or something like that
// Here The thing is performing an M step for each gmm
// It should look like :
        // for (size_t stateId = 0; stateId < hmm.nStates; stateId++) {
        //         m_step(dX, hmm.dists[stateId],
        //                cublasHandle, queue);
        //
        // }

}

}
