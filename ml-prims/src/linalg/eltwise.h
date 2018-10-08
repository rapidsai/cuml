#pragma once

#include "unary_op.h"
#include "binary_op.h"


namespace MLCommon {
namespace LinAlg {

/**
 * @defgroup ScalarOps Scalar operations on the input buffer
 * @param out the output buffer
 * @param in the input buffer
 * @param scalar the scalar used in the operations
 * @param len number of elements in the input buffer
 * @{
 */
template <typename math_t>
void scalarAdd(math_t* out, const math_t* in, math_t scalar, int len) {
    unaryOp(out, in, scalar, len, [] __device__ (math_t in, math_t scalar) {
                                      return in + scalar;
                                  });
}

template <typename math_t>
void scalarMultiply(math_t* out, const math_t* in, math_t scalar, int len) {
    unaryOp(out, in, scalar, len, [] __device__ (math_t in, math_t scalar) {
                                      return in * scalar;
                                  });
}
/** @} */


/**
 * @defgroup BinaryOps Element-wise binary operations on the input buffers
 * @param out the output buffer
 * @param in1 the first input buffer
 * @param in2 the second input buffer
 * @param len number of elements in the input buffers
 * @{
 */
template <typename math_t>
void eltwiseAdd(math_t* out, const math_t* in1, const math_t* in2, int len) {
    binaryOp(out, in1, in2, len, [] __device__ (math_t a, math_t b) {
                                     return a + b;
                                 });
}

template <typename math_t>
void eltwiseSub(math_t* out, const math_t* in1, const math_t* in2, int len) {
    binaryOp(out, in1, in2, len, [] __device__ (math_t a, math_t b) {
                                     return a - b;
                                 });
}

template <typename math_t>
void eltwiseMultiply(math_t* out, const math_t* in1, const math_t* in2, int len) {
    binaryOp(out, in1, in2, len, [] __device__ (math_t a, math_t b) {
                                     return a * b;
                                 });
}

template <typename math_t>
void eltwiseDivide(math_t* out, const math_t* in1, const math_t* in2, int len) {
    binaryOp(out, in1, in2, len, [] __device__ (math_t a, math_t b) {
                                     return a / b;
                                 });
}
/** @} */

}; // end namespace LinAlg
}; // end namespace MLCommon
