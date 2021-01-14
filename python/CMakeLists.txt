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

project(cuml VERSION ${cuml_version} LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Cython REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)
find_package(CUDAToolkit REQUIRED)

# include(cmake/CPM.cmake)
# TODO (kk): CPMFindPackage spdlog
# TODO (kk): CPMFindPackage Thrust

# Try to find an installed RMM, if not found add the parent directory
# TODO (kk): Do some basic validation to ensure the parent is as expected
# find_package(CUML ${cuml_version})
# if(NOT CUML_FOUND)
#   message(WARNING "Could not file cuML")
#   # set(BUILD_TESTS OFF)
#   # set(BUILD_BENCHMARKS OFF)
#   add_subdirectory(../cpp cuml-cpp)
# else(NOT CUML_FOUND)
#   message(WARNING "Found cuML in install DIR")
# endif(NOT CUML_FOUND)

add_subdirectory(../cpp libcuml)

include_directories(${raft_INCLUDE_DIR}/../../python)

function(add_cuml_python_libraries SOURCE_GLOB)

  # Set the list of Cython files to build
  file(GLOB cython_sources "*.pyx")

  # Build all of the Cython targets
  foreach(cython_file ${cython_sources})
    get_filename_component(cython_module_name ${cython_file} NAME_WLE)

    # Get the relative path from the py root
    file(RELATIVE_PATH cython_rel_path ${CUML_PY_ROOT} ${cython_file})
    get_filename_component(cython_module_path ${cython_rel_path} DIRECTORY)

    # Convert the relative path to a namespace. i.e. `cuml/package/module` -> `cuml::package::module
    string(REPLACE "/" "." cython_module_namespace ${cython_module_path})

    set(cython_module_target "${cython_module_namespace}.${cython_module_name}")

    # Add the target and library
    add_cython_target(${cython_module_name} "${cython_file}" CXX PY3)
    add_library(${cython_module_target} MODULE ${cython_module_name})

    set_target_properties(${cython_module_target} PROPERTIES OUTPUT_NAME "${cython_module_name}")

    # Set props, include and link
    set_target_properties(${cython_module_target} PROPERTIES PREFIX "")
    target_link_libraries(${cython_module_target} CUML::libcuml++ Python3::Module)
    target_include_directories(${cython_module_target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

    # Install
    install(TARGETS ${cython_module_target} LIBRARY DESTINATION ${cython_module_path})
  endforeach(cython_file ${cython_sources})


endfunction()

set(CUML_PY_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(cuml/cluster)
add_subdirectory(cuml/common)
add_subdirectory(cuml/datasets) # Ignored for now since arima is used in 2 places
add_subdirectory(cuml/decomposition)
add_subdirectory(cuml/ensemble)
add_subdirectory(cuml/experimental)
add_subdirectory(cuml/fil)
add_subdirectory(cuml/internals)
add_subdirectory(cuml/linear_model) # Ignored for now due to duplicate name
add_subdirectory(cuml/manifold)
add_subdirectory(cuml/metrics)
add_subdirectory(cuml/neighbors)
add_subdirectory(cuml/random_projection)
add_subdirectory(cuml/solvers)
add_subdirectory(cuml/svm)
add_subdirectory(cuml/tsa)
