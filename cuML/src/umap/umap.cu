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

    void UMAPParams::find_params_ab() {

        float step = 300 / spread*3;

        float* X = (float*)malloc(300 * sizeof(float));
        float* y = (float*)malloc(300 * sizeof(float));

        for(int i = 0; i < 300; i++) {
            X[i] = i*step;
            y[i] = 0.0;
            if(X[i] >= min_dist)
                exp(-(X[i]-min_dist)/ spread);
            else if(X[i] < min_dist)
                X[i] = 1.0;
        }

        float *X_d;
        MLCommon::allocate(X_d, 300);
        MLCommon::updateDevice(X_d, X, 300);

        float *coeffs;
        MLCommon::allocate(coeffs, 1);

        float *intercept;
        MLCommon::allocate(intercept, 1);

        Solver::sgdFit(X_d, 300, 1, y,
               coeffs, intercept, true,
               10, 5, lr_type::ADAPTIVE,
               1e-3, -1, loss_funct::SQRD_LOSS,
               MLCommon::Functions::penalty::NONE,
               -1, -1, true, 1e-3, 2);
    }
}
