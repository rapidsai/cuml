# Copyright (c) 2019, NVIDIA CORPORATION.
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
#

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
cmake_policy(SET CMP0079 NEW)

project(cuml_bench LANGUAGES CXX CUDA)

###################################################################################################
# - build ml_bench executable ----------------------------------------------------------------

if(BUILD_CUML_BENCH)

  set(ML_BENCH_LINK_LIBRARIES ${CUML_CPP_TARGET})

  # (please keep the filenames in alphabetical order)
  add_executable(bench
    sg/dataset.cpp
#    sg/dbscan.cpp
    sg/main.cpp
  )

  add_dependencies(bench ${ClangFormat_TARGET})

  target_link_libraries(bench ${ML_BENCH_LINK_LIBRARIES})

endif(BUILD_CUML_BENCH)
