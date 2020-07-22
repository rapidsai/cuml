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

# File to generate Find*.cmake modules to find packages.
# This file uses Findpkd.cmake.in to generate modules so that we can use
# find(package) in the main CMake for any package we want.
#
# Usage: The template takes the following variables:
#       PKG_NAME:
#           Name of the package to find in find(PKG_NAME)
#       PKG_HEADER_NAME:
#           Name of header file to use to find include dirs path of the package.
#       PKG_LIBRARY_NAME:
#           Name of library to find to find include lib path of the package
#       PKG_LOCATION:
#           (Optional) Name of additional folder to look for headers/lib files.
#           Useful if one wants to reduce size of #include commands
#       PKG_VERSION:
#           (Optional) Version of the package. Useful to find libraries that append
#           version number to their filename (for example libopenblas.so.0)

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

set(PKG_NAME cumlprims_mg)
set(PKG_HEADER_NAME cumlprims.hpp)
set(PKG_LIBRARY_NAME cumlprims)
set(PKG_LOCATION cumlprims)
set(PKG_VERSION 0)

configure_file(Findpkg.cmake.in
               ${CMAKE_MODULE_PATH}/Find${PKG_NAME}.cmake
               @ONLY)


set(PKG_NAME NCCL)
set(PKG_HEADER_NAME nccl.h)
set(PKG_LIBRARY_NAME nccl)
set(PKG_VERSION 2)

configure_file(Findpkg.cmake.in
               ${CMAKE_MODULE_PATH}/Find${PKG_NAME}.cmake
               @ONLY)


set(PKG_NAME RMM)
set(PKG_HEADER_NAME rmm/device_buffer.hpp)
set(PKG_LIBRARY_NAME rmm)
set(PKG_VERSION 0)

configure_file(Findpkg.cmake.in
               ${CMAKE_MODULE_PATH}/Find${PKG_NAME}.cmake
               @ONLY)


set(PKG_NAME UCX)
set(PKG_HEADER_NAME ucp/api/ucp.h)
set(PKG_LIBRARY_NAME ucp)
set(PKG_VERSION 0)

configure_file(Findpkg.cmake.in
               ${CMAKE_MODULE_PATH}/Find${PKG_NAME}.cmake
               @ONLY)




