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

#include <cutlass/gemm/dispatch.h>

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

///@todo: add support for Transform
template<cutlass::gemm::tiling_strategy::kind_t TilingStrategy,
		typename value_t, typename output_t, typename math_op,
		typename epilogue_op_t, typename dp_accum_traits_t>
struct epsneigh_dispatch {
	///@todo: support nt as well (meaning input matrix is col-major)
	static const cutlass::matrix_transform_t::kind_t TransformA =
			cutlass::matrix_transform_t::Transpose;
	static const cutlass::matrix_transform_t::kind_t TransformB =
			cutlass::matrix_transform_t::NonTranspose;

	using accum_t = value_t;
	using scalar_t = accum_t;

	static const int accumulator_alignment = sizeof(accum_t);

	/// Returns leading dimension for A matrix operand
	int leading_dim_a(int m, int k) const {
		return (TransformA == cutlass::matrix_transform_t::NonTranspose ? m : k);
	}

	/// Returns leading dimension for B matrix operand
	int leading_dim_b(int k, int n) const {
		return (TransformB == cutlass::matrix_transform_t::NonTranspose ? k : n);
	}

	/// Launches GEMM
	template<int operand_alignment>
	cutlass::gemm::launch_configuration launch(int m, int n, int k,
			epilogue_op_t epilogue_op, value_t *A, value_t *B, output_t *C,
			cudaStream_t stream = 0, bool enable_k_split = true,
			bool debug_synchronous = false) {
		return cutlass::gemm::device_gemm<TilingStrategy, math_op, TransformA,
				operand_alignment, TransformB, operand_alignment, value_t,
				accum_t, output_t, epilogue_op_t, accumulator_alignment,
				dp_accum_traits_t>(m, n, k, epilogue_op, A, B, C, stream,
				enable_k_split, debug_synchronous);
	}

	/// Dispatches CUTLASS GEMM
	cutlass::gemm::launch_configuration operator()(int m, int n, int k,
			value_t *A, value_t *B, output_t *C, epilogue_op_t epilogue,
			cudaStream_t stream = 0, bool enable_k_split = true,
			bool debug_synchronous = false) {
		const int lda = leading_dim_a(m, k);
		const int ldb = leading_dim_b(k, n);
		// Prefer the largest granularity of vector load that is compatible with
		// problem size and data alignment.
		if (!((sizeof(value_t) * lda) % 16)
				&& !((sizeof(value_t) * ldb) % 16)) {
			return launch<__NV_STD_MAX(16, sizeof(value_t))>(m, n, k, epilogue,
					A, B, C, stream, enable_k_split, debug_synchronous);
		} else if (!((sizeof(value_t) * lda) % 8)
				&& !((sizeof(value_t) * ldb) % 8)) {
			return launch<__NV_STD_MAX(8, sizeof(value_t))>(m, n, k, epilogue,
					A, B, C, stream, enable_k_split, debug_synchronous);
		} else if (!((sizeof(value_t) * lda) % 4)
				&& !((sizeof(value_t) * ldb) % 4)) {
			return launch<__NV_STD_MAX(4, sizeof(value_t))>(m, n, k, epilogue,
					A, B, C, stream, enable_k_split, debug_synchronous);
		} else if (!((sizeof(value_t) * lda) % 2)
				&& !((sizeof(value_t) * ldb) % 2)) {
			// 16b alignment is only for fp16 cases. Not yet supported here!
		}
		return cutlass::gemm::launch_configuration(cudaErrorInvalidValue);
	}
};

} // namespace Dbscan
