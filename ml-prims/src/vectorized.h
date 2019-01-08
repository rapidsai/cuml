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

#include "cuda_utils.h"
#include <cuda_fp16.h>


namespace MLCommon {


template <typename math_, int VecLen> struct IOType {};

template <> struct IOType<int8_t,1>   { typedef int8_t Type;   };
template <> struct IOType<int8_t,2>   { typedef int16_t Type;  };
template <> struct IOType<int8_t,4>   { typedef int32_t Type;  };
template <> struct IOType<int8_t,8>   { typedef int2 Type;     };
template <> struct IOType<int8_t,16>  { typedef int4 Type;     };

template <> struct IOType<uint8_t,1>  { typedef uint8_t Type;  };
template <> struct IOType<uint8_t,2>  { typedef uint16_t Type; };
template <> struct IOType<uint8_t,4>  { typedef uint32_t Type; };
template <> struct IOType<uint8_t,8>  { typedef uint2 Type;    };
template <> struct IOType<uint8_t,16> { typedef uint4 Type;    };

template <> struct IOType<int16_t,1>  { typedef int16_t Type;  };
template <> struct IOType<int16_t,2>  { typedef int32_t Type;  };
template <> struct IOType<int16_t,4>  { typedef int2 Type;     };
template <> struct IOType<int16_t,8>  { typedef int4 Type;     };

template <> struct IOType<uint16_t,1> { typedef uint16_t Type; };
template <> struct IOType<uint16_t,2> { typedef uint32_t Type; };
template <> struct IOType<uint16_t,4> { typedef uint2 Type;    };
template <> struct IOType<uint16_t,8> { typedef uint4 Type;    };

template <> struct IOType<__half,1>   { typedef __half Type;   };
template <> struct IOType<__half,2>   { typedef __half2 Type;  };
template <> struct IOType<__half,4>   { typedef uint2 Type;    };
template <> struct IOType<__half,8>   { typedef uint4 Type;    };

template <> struct IOType<__half2,1>  { typedef __half2 Type;  };
template <> struct IOType<__half2,2>  { typedef uint2 Type;    };
template <> struct IOType<__half2,4>  { typedef uint4 Type;    };

template <> struct IOType<int32_t,1>  { typedef int32_t Type;  };
template <> struct IOType<int32_t,2>  { typedef uint2 Type;    };
template <> struct IOType<int32_t,4>  { typedef uint4 Type;    };

template <> struct IOType<uint32_t,1> { typedef uint32_t Type; };
template <> struct IOType<uint32_t,2> { typedef uint2 Type;    };
template <> struct IOType<uint32_t,4> { typedef uint4 Type;    };

template <> struct IOType<float,1>    { typedef float Type;    };
template <> struct IOType<float,2>    { typedef float2 Type;   };
template <> struct IOType<float,4>    { typedef float4 Type;   };

template <> struct IOType<int64_t,1>  { typedef int64_t Type;  };
template <> struct IOType<int64_t,2>  { typedef uint4 Type;    };

template <> struct IOType<uint64_t,1> { typedef uint64_t Type; };
template <> struct IOType<uint64_t,2> { typedef uint4 Type;    };

template <> struct IOType<double,1>   { typedef double Type;   };
template <> struct IOType<double,2>   { typedef double2 Type;  };


// template <int Size> struct Cases {};

// template <> struct Cases<1> {
//     static const int arr[5] = {1, 2, 4, 8, 16};
// };
// template <> struct Cases<2> {
//     static const int arr[4] = {1, 2, 4, 8};
// };
// template <> struct Cases<4> {
//     static const int arr[3] = {1, 2, 4};
// };
// template <> struct Cases<8> {
//     static const int arr[2] = {1, 2};
// };


/**
 * @struct TxN_t
 *
 * @brief Internal data structure that is used to define a facade for vectorized
 * loads/stores across the most common POD types. The goal of his file is to
 * provide with CUDA programmers, an easy way to have compiler issue vectorized
 * load or store instructions to memory (either global or shared). Vectorized
 * accesses to memory are important as they'll utilize its resources efficiently,
 * when compared to their non-vectorized counterparts. Obviously, for whatever
 * reasons if one is unable to issue such vectorized operations, one can always
 * fallback to using POD types.
 *
 * Example demonstrating the use of load operations, performing math on such
 * loaded data and finally storing it back.
 * <pre>
 * TxN_t<uint8_t,8> mydata1, mydata2;
 * int idx = (threadIdx.x + (blockIdx.x * blockDim.x)) * mydata1.Ratio;
 * mydata1.load(ptr1, idx);
 * mydata2.load(ptr2, idx);
 * #pragma unroll
 * for(int i=0;i<mydata1.Ratio;++i) {
 *     mydata1.val.data[i] += mydata2.val.data[i];
 * }
 * mydata1.store(ptr1, idx);
 * </pre>
 *
 * By doing as above, the interesting thing is that the code effectively remains
 * almost the same, in case one wants to upgrade to TxN_t<uint16_t,16> type.
 * Only change required is to replace variable declaration appropriately.
 *
 * Obviously, it's caller's responsibility to take care of pointer alignment!
 *
 * @tparam math_ the data-type in which the compute/math needs to happen
 * @tparam veclen_ the number of 'math_' types to be loaded/stored per instruction
 */
template <typename math_, int veclen_>
struct TxN_t {
    /** underlying math data type */
    typedef math_ math_t;
    /** internal storage data type */
    typedef typename IOType<math_t,veclen_>::Type io_t;

    /** defines the number of 'math_t' types stored by this struct */
    static const int Ratio = veclen_;

    union {
        /** the vectorized data that is used for subsequent operations */
        math_t data[Ratio];
        /** internal data used to ensure vectorized loads/stores */
        io_t internal;
    } val;

    ///@todo: add default constructor

    /**
     * @brief Fill the contents of this structure with a constant value
     * @param _val the constant to be filled
     */
    DI void fill(math_t _val) {
        #pragma unroll
        for(int i=0;i<Ratio;++i) {
            val.data[i] = val;
        }
    }

    ///@todo: how to handle out-of-bounds!!?

    /**
     * @defgroup LoadsStores Global/Shared vectored loads or stores
     *
     * @brief Perform vectored loads/stores on this structure
     *
     * @param ptr base pointer from where to load (or store) the data. It must
     *  be aligned to 'sizeof(io_t)'!
     * @param idx the offset from the base pointer which will be loaded
     *  (or stored) by the current thread. This must be aligned to 'Ratio'!
     *
     * @note: In case of loads, after a successful execution, the val.data will
     *  be populated with the desired data loaded from the pointer location. In
     * case of stores, the data in the val.data will be stored to that location.
     * @{
     */
    DI void load(const math_t* ptr, int idx) {
        const io_t* bptr = reinterpret_cast<const io_t*>(&ptr[idx]);
        val.internal = __ldg(bptr);
    }

    DI void load(math_t* ptr, int idx) {
        io_t* bptr = reinterpret_cast<io_t*>(&ptr[idx]);
        val.internal = *bptr;
    }

    DI void store(math_t* ptr, int idx) {
        io_t* bptr = reinterpret_cast<io_t*>(&ptr[idx]);
        *bptr = val.internal;
    }
    /** @} */
};


/** this is just to keep the compiler happy! */
template <typename math_>
struct TxN_t<math_,0> {
    typedef math_ math_t;
    static const int Ratio = 1;

    union {
        math_t data[1];
    } val;

    DI void fill(math_t _val) { }
    DI void load(const math_t* ptr, int idx) { }
    DI void load(math_t* ptr, int idx) { }
    DI void store(math_t* ptr, int idx) { }
};

}; // namespace MLCommon
