#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
project(cuML-comms-MPI LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)

if(NOT NCCL_PATH)
  find_package(NCCL REQUIRED)
else()
  set(NCCL_INCLUDE_DIRS ${NCCL_PATH}/include)
  set(NCCL_LIBRARIES ${NCCL_PATH}/lib/libnccl.so)
  set(NCCL_FOUND ON)
endif(NOT NCCL_PATH)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${MPI_CXX_INCLUDE_PATH}
  ../../include
  ../../src
  ../../src_prims
)

set(MPI_COMMS_LINK_LIBRARIES ${CUML_CPP_TARGET} ${MPI_C_LIBRARIES})

if (NCCL_FOUND)
    add_definitions(-DHAVE_NCCL)
    include_directories( ${NCCL_INCLUDE_DIRS} )
    list(APPEND MPI_COMMS_LINK_LIBRARIES ${NCCL_LIBRARIES})
endif()

add_library(cumlcommsmpi SHARED src/cuML_comms_mpi_impl.cpp)
target_link_libraries(cumlcommsmpi ${MPI_COMMS_LINK_LIBRARIES})
target_compile_options(cumlcommsmpi PUBLIC ${MPI_C_COMPILE_FLAGS})

install(TARGETS cumlcommsmpi DESTINATION lib)
