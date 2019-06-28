#
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

cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(cuML-comms VERSION 0.8.0 LANGUAGES CXX CUDA)

set(CUML_DIR ${PROJECT_SOURCE_DIR}/.. CACHE STRING "Path to the cuML repo")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CUML_DIR}/cmake")

option(WITH_UCX "Uses UCX for p2p comms" OFF)


find_package(NCCL REQUIRED)
find_package(CUDA 9.0 REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(_CUML_LIBRARY_HINTS ${CUML_DIR})
if (CUML_INSTALL_DIR)
    list(APPEND _CUML_LIBRARY_HINTS ${CUML_INSTALL_DIR})
endif()
find_library(_CUML_LIBRARY NAMES libcuml++ cuml++ HINTS ${_CUML_LIBRARY_HINTS} PATH_SUFFIXES lib build)

set(CUML_LIBRARY ${_CUML_LIBRARY} CACHE STRING "Path to the cuML lib")
unset(_CUML_LIBRARY)
unset(_CUML_LIBRARY_HINTS)

add_compile_definitions(WITH_UCX=${WITH_UCX})

include_directories( include 
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} 
	${CUML_DIR}/src 
	${CUML_DIR}/src_prims 
	${UCX_PATH}/include
)

set(CUML_LINK_LIBRARIES ${CUML_LIBRARY})

if(WITH_UCX)
set(CUML_LINK_LIBRARIES ${CUML_LINK_LIBRARIES} ucp uct ucm ucs pthread)
endif(WITH_UCX)

add_definitions(-DHAVE_NCCL)
include_directories( ${NCCL_INCLUDE_DIRS} )
list(APPEND CUML_LINK_LIBRARIES ${NCCL_LIBRARIES})

add_library(cumlcomms SHARED src/cuML_std_comms_impl.cpp)
target_link_libraries(cumlcomms ${CUML_LINK_LIBRARIES})

install(TARGETS cumlcomms DESTINATION lib)
