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
# To use, include in CMakeLists.txt this folder, and then use the function
# GENERATE_FIND_MODULE, which takes the following parameters:
#       NAME:
#           Name of the package to find in find(NAME)
#       HEADER_NAME:
#           Name of header file to use to find include dirs path of the package.
#       LIBRARY_NAME:
#           (Optional) Name of library to find to find include lib path of the package.
#           Assumed to be lib${NAME} if left empty.
#       LOCATION:
#           (Optional) Name of additional folder to look for headers/lib files.
#           Useful if one wants to reduce size of #include commands
#       VERSION:
#           (Optional) Version of the package. Useful to find libraries that append
#           version number to their filename (for example libopenblas.so.0)


function(GENERATE_FIND_MODULE)
    set(oneValueArgs NAME LOCATION VERSION)
    set(multiValueArgs HEADER_NAME LIBRARY_NAME)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    if(NOT DEFINED PKG_LIBRARY_NAME)
        set(PKG_LIBRARY_NAME ${PKG_NAME})
    endif(NOT DEFINED PKG_LIBRARY_NAME)

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/Findpkg.cmake.in
               ${CMAKE_CURRENT_BINARY_DIR}/cmake/Find${PKG_NAME}.cmake
               @ONLY)
endfunction()

