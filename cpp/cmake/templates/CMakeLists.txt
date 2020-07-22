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

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

set(PKG_NAME cumlprims_mg)
set(PKG_HEADER_NAME cumlprims/cumlprims.hpp)
set(PKG_LIBRARY_NAME cumlprims)
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
set(PKG_HEADER_NAME device_buffer.hpp)
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




