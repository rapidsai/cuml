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

namespace ML {

    template<typename T>
    class UMAPState {
        public:
            T *graph_vals;
            int *graph_rows;
            int *graph_cols;
    };

	class UMAPParams {
	public:

	    int n_neighbors = 15;
		int n_components = 2;
		int n_epochs = -1;

		float learning_rate = 1.0;
		float min_dist = 0.1;
		float spread = 1.0;
		float set_op_mix_ratio = 1.0;

		int local_connectivity = 1;
		float repulsion_strength = 1.0;
		int negative_sample_rate = 5;
		float transform_queue_size = 4.0;


		/**
		 * Parameters of differentiable approx
		 * of right adjoint functor.
		 */
		float a, b;

		float gamma;

		int target_n_neighbors = -1;
		float target_weight = 0.5;

		/**
		 * Initial learning rate for SGD
		 */
		float initial_alpha = 1.0;

		/**
		 * Fit a, b params for the differentiable curve used in
		 * lower dimensional fuzzy simplicial complex construction.
		 * We want the smooth curve (from a pre-defined family with
		 * simple gradient) that best matches an offset exponential
		 * decay.
		 */
	};


	template<class T>
	class UMAP {

	private:
		UMAPParams *params;
		UMAPState<T> *state;

	public:

		UMAP(UMAPParams *params): params(params), state(new UMAPState<T>()){}

		void fit(T *X, int n, int d);

		void transform(T *X);

		UMAPParams* get_params();

		UMAPState<T>* get_state();
	};

}

