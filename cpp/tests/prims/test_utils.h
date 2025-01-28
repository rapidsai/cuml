/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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
#include <raft/core/interruptible.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <memory>

namespace MLCommon {

template <typename T>
struct Compare {
  bool operator()(const T& a, const T& b) const { return a == b; }
};

template <typename T>
struct CompareApprox {
  CompareApprox(T eps_) : eps(eps_) {}
  bool operator()(const T& a, const T& b) const
  {
    T diff  = abs(a - b);
    T m     = std::max(abs(a), abs(b));
    T ratio = diff >= eps ? diff / m : diff;

    return (ratio <= eps);
  }

 private:
  T eps;
};

template <typename T>
struct CompareApproxAbs {
  CompareApproxAbs(T eps_) : eps(eps_) {}
  bool operator()(const T& a, const T& b) const
  {
    T diff  = abs(abs(a) - abs(b));
    T m     = std::max(abs(a), abs(b));
    T ratio = diff >= eps ? diff / m : diff;
    return (ratio <= eps);
  }

 private:
  T eps;
};

template <typename T>
HDI T abs(const T& a)
{
  return a > T(0) ? a : -a;
}

/*
 * @brief Helper function to compare 2 device n-D arrays with custom comparison
 * @tparam T the data type of the arrays
 * @tparam L the comparator lambda or object function
 * @param expected expected value(s)
 * @param actual actual values
 * @param eq_compare the comparator
 * @param stream cuda stream
 * @return the testing assertion to be later used by ASSERT_TRUE/EXPECT_TRUE
 * @{
 */
template <typename T, typename L>
testing::AssertionResult devArrMatch(
  const T* expected, const T* actual, size_t size, L eq_compare, cudaStream_t stream = 0)
{
  std::unique_ptr<T[]> exp_h(new T[size]);
  std::unique_ptr<T[]> act_h(new T[size]);
  raft::update_host<T>(exp_h.get(), expected, size, stream);
  raft::update_host<T>(act_h.get(), actual, size, stream);
  raft::interruptible::synchronize(stream);
  for (size_t i(0); i < size; ++i) {
    auto exp = exp_h.get()[i];
    auto act = act_h.get()[i];
    if (!eq_compare(exp, act)) {
      return testing::AssertionFailure() << "actual=" << act << " != expected=" << exp << " @" << i;
    }
  }
  return testing::AssertionSuccess();
}

template <typename T, typename L>
testing::AssertionResult devArrMatch(
  T expected, const T* actual, size_t size, L eq_compare, cudaStream_t stream = 0)
{
  std::unique_ptr<T[]> act_h(new T[size]);
  raft::update_host<T>(act_h.get(), actual, size, stream);
  raft::interruptible::synchronize(stream);
  for (size_t i(0); i < size; ++i) {
    auto act = act_h.get()[i];
    if (!eq_compare(expected, act)) {
      return testing::AssertionFailure()
             << "actual=" << act << " != expected=" << expected << " @" << i;
    }
  }
  return testing::AssertionSuccess();
}

template <typename T, typename L>
testing::AssertionResult devArrMatch(const T* expected,
                                     const T* actual,
                                     size_t rows,
                                     size_t cols,
                                     L eq_compare,
                                     cudaStream_t stream = 0)
{
  size_t size = rows * cols;
  std::unique_ptr<T[]> exp_h(new T[size]);
  std::unique_ptr<T[]> act_h(new T[size]);
  raft::update_host<T>(exp_h.get(), expected, size, stream);
  raft::update_host<T>(act_h.get(), actual, size, stream);
  raft::interruptible::synchronize(stream);
  for (size_t i(0); i < rows; ++i) {
    for (size_t j(0); j < cols; ++j) {
      auto idx = i * cols + j;  // row major assumption!
      auto exp = exp_h.get()[idx];
      auto act = act_h.get()[idx];
      if (!eq_compare(exp, act)) {
        return testing::AssertionFailure()
               << "actual=" << act << " != expected=" << exp << " @" << i << "," << j;
      }
    }
  }
  return testing::AssertionSuccess();
}

template <typename T, typename L>
testing::AssertionResult devArrMatch(
  T expected, const T* actual, size_t rows, size_t cols, L eq_compare, cudaStream_t stream = 0)
{
  size_t size = rows * cols;
  std::unique_ptr<T[]> act_h(new T[size]);
  raft::update_host<T>(act_h.get(), actual, size, stream);
  raft::interruptible::synchronize(stream);
  for (size_t i(0); i < rows; ++i) {
    for (size_t j(0); j < cols; ++j) {
      auto idx = i * cols + j;  // row major assumption!
      auto act = act_h.get()[idx];
      if (!eq_compare(expected, act)) {
        return testing::AssertionFailure()
               << "actual=" << act << " != expected=" << expected << " @" << i << "," << j;
      }
    }
  }
  return testing::AssertionSuccess();
}

/*
 * @brief Helper function to compare a device n-D arrays with an expected array
 * on the host, using a custom comparison
 * @tparam T the data type of the arrays
 * @tparam L the comparator lambda or object function
 * @param expected_h host array of expected value(s)
 * @param actual_d device array actual values
 * @param eq_compare the comparator
 * @param stream cuda stream
 * @return the testing assertion to be later used by ASSERT_TRUE/EXPECT_TRUE
 */
template <typename T, typename L>
testing::AssertionResult devArrMatchHost(
  const T* expected_h, const T* actual_d, size_t size, L eq_compare, cudaStream_t stream = 0)
{
  std::unique_ptr<T[]> act_h(new T[size]);
  raft::update_host<T>(act_h.get(), actual_d, size, stream);
  raft::interruptible::synchronize(stream);
  bool ok   = true;
  auto fail = testing::AssertionFailure();
  for (size_t i(0); i < size; ++i) {
    auto exp = expected_h[i];
    auto act = act_h.get()[i];
    if (!eq_compare(exp, act)) {
      ok = false;
      fail << "actual=" << act << " != expected=" << exp << " @" << i << "; ";
    }
  }
  if (!ok) return fail;
  return testing::AssertionSuccess();
}

/*
 * @brief Helper function to compare diagonal values of a 2D matrix
 * @tparam T the data type of the arrays
 * @tparam L the comparator lambda or object function
 * @param expected expected value along diagonal
 * @param actual actual matrix
 * @param eq_compare the comparator
 * @param stream cuda stream
 * @return the testing assertion to be later used by ASSERT_TRUE/EXPECT_TRUE
 */
template <typename T, typename L>
testing::AssertionResult diagonalMatch(
  T expected, const T* actual, size_t rows, size_t cols, L eq_compare, cudaStream_t stream = 0)
{
  size_t size = rows * cols;
  std::unique_ptr<T[]> act_h(new T[size]);
  raft::update_host<T>(act_h.get(), actual, size, stream);
  raft::interruptible::synchronize(stream);
  for (size_t i(0); i < rows; ++i) {
    for (size_t j(0); j < cols; ++j) {
      if (i != j) continue;
      auto idx = i * cols + j;  // row major assumption!
      auto act = act_h.get()[idx];
      if (!eq_compare(expected, act)) {
        return testing::AssertionFailure()
               << "actual=" << act << " != expected=" << expected << " @" << i << "," << j;
      }
    }
  }
  return testing::AssertionSuccess();
}

template <typename T, typename L>
testing::AssertionResult match(const T expected, T actual, L eq_compare)
{
  if (!eq_compare(expected, actual)) {
    return testing::AssertionFailure() << "actual=" << actual << " != expected=" << expected;
  }
  return testing::AssertionSuccess();
}

/** @} */

/** time the function call 'func' using cuda events */
#define TIMEIT_LOOP(ms, count, func)                       \
  do {                                                     \
    cudaEvent_t start, stop;                               \
    RAFT_CUDA_TRY(cudaEventCreate(&start));                \
    RAFT_CUDA_TRY(cudaEventCreate(&stop));                 \
    RAFT_CUDA_TRY(cudaEventRecord(start));                 \
    for (int i = 0; i < count; ++i) {                      \
      func;                                                \
    }                                                      \
    RAFT_CUDA_TRY(cudaEventRecord(stop));                  \
    RAFT_CUDA_TRY(cudaEventSynchronize(stop));             \
    ms = 0.f;                                              \
    RAFT_CUDA_TRY(cudaEventElapsedTime(&ms, start, stop)); \
    ms /= args.runs;                                       \
  } while (0)

};  // namespace MLCommon
