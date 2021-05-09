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

function(find_and_configure_gbench VERSION PINNED_TAG)

    # if(TARGET benchmark::benchmark)
    #     return()
    # endif()

    # set(oneValueArgs VERSION PINNED_TAG)
    # cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
    #                       "${multiValueArgs}" ${ARGN} )

    # rapids_cpm_find(benchmark ${PKG_VERSION}
    #     GLOBAL_TARGETS      benchmark::benchmark
    #     CPM_ARGS
    #         GIT_REPOSITORY  https://github.com/google/benchmark.git
    #         GIT_TAG         ${PKG_PINNED_TAG}
    #         GIT_SHALLOW     TRUE
    #         OPTIONS
    #           "DBENCHMARK_ENABLE_GTEST_TESTS OFF"
    #           "DBENCHMARK_ENABLE_TESTING OFF"
    #           "DCMAKE_INSTALL_PREFIX <INSTALL_DIR>"
    #           "DCMAKE_BUILD_TYPE Release"
    #           "DCMAKE_INSTALL_LIBDIR lib"
    # )

    # # if(NOT TARGET benchmark::benchmark)
    # #     add_library(benchmark::benchmark ALIAS benchmark)
    # # endif()

    # macro(print_all_variables)
    #     message(STATUS "print_all_variables------------------------------------------{")
    #     get_cmake_property(_variableNames VARIABLES)
    #     foreach (_variableName ${_variableNames})
    #         message(STATUS "${_variableName}=${${_variableName}}")
    #     endforeach()
    #     message(STATUS "print_all_variables------------------------------------------}")
    # endmacro()

    # print_all_variables()

    set(GBENCH_DIR ${CMAKE_CURRENT_BINARY_DIR}/benchmark CACHE STRING
      "Path to google benchmark repo")
    set(GBENCH_BINARY_DIR ${PROJECT_BINARY_DIR}/benchmark)
    set(GBENCH_INSTALL_DIR ${GBENCH_BINARY_DIR}/install)
    set(GBENCH_LIB ${GBENCH_INSTALL_DIR}/lib/libbenchmark.a)
    include(ExternalProject)
    ExternalProject_Add(benchmark
      GIT_REPOSITORY    https://github.com/google/benchmark.git
      GIT_TAG           bf4f2ea0bd1180b34718ac26eb79b170a4f6290e
      PREFIX            ${GBENCH_DIR}
      CMAKE_ARGS        -DBENCHMARK_ENABLE_GTEST_TESTS=OFF
                        -DBENCHMARK_ENABLE_TESTING=OFF
                        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                        -DCMAKE_BUILD_TYPE=Release
                        -DCMAKE_INSTALL_LIBDIR=lib
      BUILD_BYPRODUCTS  ${GBENCH_DIR}/lib/libbenchmark.a
      UPDATE_COMMAND    "")
    add_library(benchmark::benchmark STATIC IMPORTED)
    add_dependencies(benchmark::benchmark benchmark)
    set_property(TARGET benchmark::benchmark PROPERTY
      IMPORTED_LOCATION ${GBENCH_DIR}/lib/libbenchmark.a)

endfunction()

find_and_configure_gbench(VERSION 1.5.1
                          PINNED_TAG 70d89ac5190923bdaebf9d0a4ac04085796d062a)
