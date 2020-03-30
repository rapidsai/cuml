#=============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

include(ExternalProject)

set(CUB_DIR ${CMAKE_CURRENT_BINARY_DIR}/cub CACHE STRING "Path to cub repo")
set(CUB_VERSION v1.8.0 CACHE STRING "cub branch version to use")
ExternalProject_Add(cub
  GIT_REPOSITORY    https://github.com/NVlabs/cub.git
  GIT_TAG           ${CUB_VERSION}
  PREFIX            ${CUB_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
)

set(CUTLASS_DIR ${CMAKE_CURRENT_BINARY_DIR}/cutlass CACHE STRING
  "Path to the cutlass repo")
ExternalProject_Add(cutlass
  GIT_REPOSITORY    https://github.com/NVIDIA/cutlass.git
  GIT_TAG           v1.0.1
  PREFIX            ${CUTLASS_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
)

# dependencies will be added in sequence, so if a new project `project_b` is added
# after `project_a`, please add the dependency add_dependencies(project_b project_a)
# This allows the cloning to happen sequentially, enhancing the printing at
# compile time, helping significantly to troubleshoot build issues.
add_dependencies(cutlass cub)
