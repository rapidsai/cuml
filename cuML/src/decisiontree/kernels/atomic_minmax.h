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

__device__ __forceinline__ float atomicMinFD(float * addr, float value) {

	float old;
	old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
		__uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
	
	return old;
}

__device__ __forceinline__ float atomicMaxFD(float * addr, float value) {

	float old;
	old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
		__uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));
	
	return old;
}

//FIXME TODO: Test the double overloaded impl.
__device__ __forceinline__ double atomicMaxFD(double* address, double val) {
	
    unsigned long long* address_as_ull = (unsigned long long *) address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmaxf(val, __longlong_as_double(assumed))));
    } while (assumed != old);
	
    return __longlong_as_double(old);
}
	

	
__device__ __forceinline__ double atomicMinFD(double* address, double val) {
	
    unsigned long long* address_as_ull = (unsigned long long*) address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fminf(val, __longlong_as_double(assumed))));
    } while (assumed != old);
	
    return __longlong_as_double(old);
}
