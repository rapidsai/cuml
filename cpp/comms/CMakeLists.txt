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

find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUML_DIR ${PROJECT_SOURCE_DIR}/.. CACHE STRING "Path to the cuML repo")
set(CUML_LIBRARY ${CUML_DIR}/build/libcuml++.so CACHE STRING "Path to the cuML lib")
set(NCCL_INCLUDE_DIR /home/jkraus/workspace/NCCL/nccl/build/include CACHE STRING "NCCL include path.")
set(NCCL_LIBRARY /home/jkraus/workspace/NCCL/nccl/build/lib/libnccl.so CACHE STRING "Path to the NCCL library.")

add_definitions(-DHAVE_NCCL)

include_directories( include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${MPI_CXX_INCLUDE_PATH} ${CUML_DIR}/src ${CUML_DIR}/src_prims ${NCCL_INCLUDE_DIR} )

add_library(cumlcomms SHARED src/cuML_comms_mpi_impl.cpp)
target_link_libraries(cumlcomms ${CUML_LIBRARY} ${MPI_C_LIBRARIES})
target_compile_options(cumlcomms PUBLIC ${MPI_C_COMPILE_FLAGS})
