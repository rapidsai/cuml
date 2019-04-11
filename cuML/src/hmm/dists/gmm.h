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

namespace gmm {

template <typename T>
size_t get_workspace_size(GMM<T> &gmm){
        size_t workspaceSize = gmm_bufferSize(gmm);
        return workspaceSize;
}

template <typename T>
void create_handle(GMM<T> &gmm, void* workspace){
        create_GMMHandle_new(gmm, workspace);
}

template <typename T>
void update_emissions(hmm::HMM<T, GMM<T> > &hmm,
                      T* dX){

}


}
