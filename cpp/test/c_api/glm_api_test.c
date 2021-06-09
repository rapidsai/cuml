/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cuml/linear_model/glm_api.h>

void test_glm() {

   cumlHandle_t handle = 0;
   cumlError_t response = CUML_SUCCESS;

   response = cumlSpQnFit(handle, NULL, NULL, 0, 1, 2, false, 1.0f, 2.0f, 3, 3.0f, 3.5f, 4, 5, 6, NULL, NULL, NULL, true, 7);

   response = cumlDpQnFit(handle, NULL, NULL, 0, 1, 2, false, 1.0f, 2.0f, 3, 3.0f, 3.5f, 4, 5, 6, NULL, NULL, NULL, true, 7);

}
