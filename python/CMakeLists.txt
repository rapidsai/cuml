#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.18...3.18)

set(cuml_version 0.18.0)

project(cuml-python VERSION ${cuml_version} LANGUAGES CXX CUDA)

find_package(Cython REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)
find_package(CUDAToolkit REQUIRED)

# include(cmake/CPM.cmake)
# TODO (kk): CPMFindPackage spdlog
# TODO (kk): CPMFindPackage Thrust

# Try to find an installed RMM, if not found add the parent directory
# TODO (kk): Do some basic validation to ensure the parent is as expected
find_package(cuML ${cuml_version})
if(NOT cuML_FOUND)
  set(BUILD_TESTS OFF)
  set(BUILD_BENCHMARKS OFF)
  add_subdirectory(../cpp cuml-cpp)
endif(NOT cuML_FOUND)

add_subdirectory(cuml/cluster)
