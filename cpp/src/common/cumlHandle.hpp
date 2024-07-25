/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include <cuml/cuml_api.h>

#include <raft/core/handle.hpp>

namespace ML {

/**
 * Map from integral cumlHandle_t identifiers to cumlHandle pointer protected
 * by a mutex for thread-safe access.
 */
class HandleMap {
 public:
  /**
   * @brief Creates new handle object with associated handle ID and insert into map.
   *
   * @param[in] stream the stream to which cuML work should be ordered.
   * @return std::pair with handle and error code. If error code is not CUML_SUCCESS
   *                   the handle is INVALID_HANDLE.
   */
  std::pair<cumlHandle_t, cumlError_t> createAndInsertHandle(cudaStream_t stream);

  /**
   * @brief Lookup pointer to handle object for handle ID in map.
   *
   * @return std::pair with handle and error code. If error code is not CUML_SUCCESS
   *                   the handle is INVALID_HANDLE. Error code CUML_INAVLID_HANDLE
   *                   is returned if the provided `handle` is invalid.
   */
  std::pair<raft::handle_t*, cumlError_t> lookupHandlePointer(cumlHandle_t handle) const;

  /**
   * @brief Remove handle from map and destroy associated handle object.
   *
   * @return cumlError_t CUML_SUCCESS or CUML_INVALID_HANDLE.
   *                   Error code CUML_INAVLID_HANDLE is returned if the provided
   *                   `handle` is invalid.
   */
  cumlError_t removeAndDestroyHandle(cumlHandle_t handle);

  static const cumlHandle_t INVALID_HANDLE = -1;  //!< sentinel value for invalid ID

 private:
  std::unordered_map<cumlHandle_t, raft::handle_t*> _handleMap;  //!< map from ID to pointer
  mutable std::mutex _mapMutex;                                  //!< mutex protecting the map
  cumlHandle_t _nextHandle;                                      //!< value of next handle ID
};

/// Static handle map instance (see cumlHandle.cpp)
extern HandleMap handleMap;

}  // end namespace ML
