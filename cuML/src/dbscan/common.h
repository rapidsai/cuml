/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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


namespace Dbscan {

/// Default "ds1" diff-squared-accumulate traits specialization for value_t->accum_t
/// Currently only supported for float/double
template<typename value_t, typename accum_t>
struct ds_accummulate {
	/// Single-component "ds1" diff-squared vector type
	typedef value_t dp_vector_t;

	/// Compute "ds1" float->float
	inline __device__ static void mad(float &d, const float &a, const float &b,
			const float &c) {
		float diff = a - b;
		asm volatile ( "fma.rn.f32 %0, %1, %1, %2;\n"
				: "=f"(d) : "f"(diff), "f"(c));
	}

	/// Compute "ds1" double->double
	inline __device__ static void mad(double &d, const double &a,
			const double &b, const double &c) {
		double diff = a - b;
		asm volatile ("fma.rn.f64 %0, %1, %1, %2;\n"
				: "=d"(d) : "d"(diff), "d"(c));
	}
};

} // namespace Dbscan
