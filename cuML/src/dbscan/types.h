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

template<typename Type>
struct Type4 {
	Type a, b, c, d;
};

template<>
struct __align__(16) Type4<float> {
	float a, b, c, d;
};

template <typename Type>
DI Type4<Type> operator-(const Type4<Type>& x, const Type4<Type>& y) {
	Type4<Type> res;
	res.a = x.a - y.a;
	res.b = x.b - y.b;
	res.c = x.c - y.c;
	res.d = x.d - y.d;
	return res;
}

template <typename Type>
DI void sts(int32_t addr, Type x) {
}

template <>
DI void sts<float>(int32_t addr, float x) {
	asm volatile("st.volatile.shared.f32 [%0], %1;" : : "r"(addr), "f"(x));
}

template <typename Type>
DI void lds(Type4<Type>& x, int32_t addr) {
}

template <>
DI void lds<float>(Type4<float>& x, int32_t addr) {
	asm volatile("ld.volatile.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
			: "=f"(x.a), "=f"(x.b), "=f"(x.c), "=f"(x.d)
			: "r"(addr));
}

}
 // namespace Dbscan
