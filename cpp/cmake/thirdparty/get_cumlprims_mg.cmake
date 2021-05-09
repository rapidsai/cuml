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

function(find_and_configure_cumlprims_mg)

    if(TARGET cumlprims_mg::cumlprims_mg)
        return()
    endif()

    set(oneValueArgs VERSION)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    rapids_find_generate_module(cumlprims_mg
        HEADER_NAMES          cumlprims.hpp
        LIBRARY_NAMES         cumlprims
        INCLUDE_SUFFIXES      cumlprims
    )

    rapids_find_package(cumlprims_mg REQUIRED)

endfunction()

set(CUML_MIN_VERSION_cumlprims_mg "${CUML_VERSION_MAJOR}.${CUML_VERSION_MINOR}")

find_and_configure_cumlprims_mg(VERSION     CUML_MIN_VERSION_cumlprims_mg)

