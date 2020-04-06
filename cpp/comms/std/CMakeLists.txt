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

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(cuML-comms LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CUML_DIR}/cmake")

option(WITH_UCX "Uses UCX for P2P comms" ON)

if(NOT NCCL_PATH)
    find_package(NCCL REQUIRED)
else()
    message("-- Manually set NCCL PATH to ${NCCL_PATH}")
    set(NCCL_INCLUDE_DIRS ${NCCL_PATH}/include)
    set(NCCL_LIBRARIES ${NCCL_PATH}/lib/libnccl.so)
endif(NOT NCCL_PATH)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ../../include
  ../../src
  ../../src_prims
)

set(COMMS_LINK_LIBRARIES ${CUML_CPP_TARGET})

# Note this option will be removed once UCX conda package is released
if(WITH_UCX)
    # dlopen is used to dynamically load the needed ucp symbols at runtime. 
    # Only the UCX include directories are needed for compiling
  	find_package(UCX)
	include_directories(${UCX_INCLUDE_DIRS})
	add_compile_definitions(WITH_UCX=1)
endif(WITH_UCX)

add_definitions(-DHAVE_NCCL)
include_directories( ${NCCL_INCLUDE_DIRS} )
list(APPEND COMMS_LINK_LIBRARIES ${NCCL_LIBRARIES})

add_library(cumlcomms SHARED src/cuML_std_comms_impl.cpp)
target_link_libraries(cumlcomms ${COMMS_LINK_LIBRARIES})

install(TARGETS cumlcomms DESTINATION lib)
