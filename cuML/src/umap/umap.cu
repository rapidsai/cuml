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

#include "umap.h"
#include "optimize.h"
#include "runner.h"

#include "ml_utils.h"

#include "solver/solver_c.h"
#include "solver/learning_rate.h"
#include "functions/penalty.h"
#include "functions/linearReg.h"

namespace ML {

    using namespace UMAPAlgo;



	/***
	 * Fit a UMAP model, currently completely unsupervised.
	 */
    template<class T>
	void UMAP<T>::fit(T *X, int n, int d) {
		_fit(X, n, d, get_params(), get_state());
	}

    template<class T>
	void UMAP<T>::transform(T *x) {

	}

    template<class T>
	UMAPParams* UMAP<T>::get_params() { return this->params; }

    template<class T>
    UMAPState<T>* UMAP<T>::get_state() { return this->state; }

}
