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

function(find_and_configure_gbench)

    set(oneValueArgs VERSION PINNED_TAG)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    rapids_cpm_find(benchmark ${PKG_VERSION}
        GLOBAL_TARGETS benchmark::benchmark
        CPM_ARGS
            GIT_REPOSITORY  https://github.com/google/benchmark.git
            GIT_TAG         ${PKG_PINNED_TAG}
            OPTIONS
              "BENCHMARK_ENABLE_GTEST_TESTS OFF"
              "BENCHMARK_ENABLE_TESTING OFF"
              "BENCHMARK_ENABLE_INSTALL OFF"
              "CMAKE_BUILD_TYPE Release"
              "CMAKE_INSTALL_LIBDIR lib"
    )

    if(NOT TARGET benchmark::benchmark)
        add_library(benchmark::benchmark ALIAS benchmark)
    endif()

endfunction()

find_and_configure_gbench(VERSION      1.5.3
                          PINNED_TAG   c05843a9f622db08ad59804c190f98879b76beba)
