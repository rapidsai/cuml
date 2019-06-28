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

#ifdef DEBUG
#define COUT() (std::cout)
#define CERR() (std::cerr)

#define WARNING(message)                                                       \
  do {                                                                         \
    std::stringstream ss;                                                      \
    ss << "Warning (" << __FILE__ << ":" << __LINE__ << "): " << message;      \
    CERR() << ss.str() << std::endl;                                           \
  } while (0)

#define CASE_STR(CODE)                                                         \
    case CODE:                                                                 \
        CERR() << #CODE << std::endl;                                          \
        break
#define CHECK_CUBLAS(call)                                                     \
  {                                                                            \
    switch (call) {                                                            \
    case CUBLAS_STATUS_SUCCESS:                                                \
      break;                                                                   \
      CASE_STR(CUBLAS_STATUS_NOT_INITIALIZED);                                 \
      CASE_STR(CUBLAS_STATUS_ALLOC_FAILED);                                    \
      CASE_STR(CUBLAS_STATUS_INVALID_VALUE);                                   \
      CASE_STR(CUBLAS_STATUS_ARCH_MISMATCH);                                   \
      CASE_STR(CUBLAS_STATUS_MAPPING_ERROR);                                   \
      CASE_STR(CUBLAS_STATUS_EXECUTION_FAILED);                                \
      CASE_STR(CUBLAS_STATUS_INTERNAL_ERROR);                                  \
    default: CERR() << "unknown CUBLAS error" << std::endl;                    \
    }                                                                          \
  }

#define CHECK_CUSOLVER(call)                                                   \
  {                                                                            \
    switch (call) {                                                            \
    case CUSOLVER_STATUS_SUCCESS:                                              \
      break;                                                                   \
      CASE_STR(CUSOLVER_STATUS_NOT_INITIALIZED);                               \
      CASE_STR(CUSOLVER_STATUS_ALLOC_FAILED);                                  \
      CASE_STR(CUSOLVER_STATUS_INVALID_VALUE);                                 \
      CASE_STR(CUSOLVER_STATUS_ARCH_MISMATCH);                                 \
      CASE_STR(CUSOLVER_STATUS_EXECUTION_FAILED);                              \
      CASE_STR(CUSOLVER_STATUS_INTERNAL_ERROR);                                \
      CASE_STR(CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED);                     \
    default: CERR() << "unknown CUSOLVER error" << std::endl;                  \
    }                                                                          \
  }

#else  // DEBUG
  #define WARNING(message)
  #define CHECK_CUBLAS(call) (call)
  #define CHECK_CUSOLVER(call) (call)
#endif
