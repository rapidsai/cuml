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
#

# Based on FindPNG.cmake from cmake 3.14.3

#[=======================================================================[.rst:
FindRMM
--------

Template to generate FindPKG_NAME.cmake CMake modules

Find RMM

Imported targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` target:

``RMM::RMM``
  The libRMM library, if found.

Result variables
^^^^^^^^^^^^^^^^

This module will set the following variables in your project:

``RMM_INCLUDE_DIRS``
  where to find RMM.hpp , etc.
``RMM_LIBRARIES``
  the libraries to link against to use libRMM.
``RMM_FOUND``
  If false, do not try to use RMM.
``RMM_VERSION_STRING``
  the version of the RMM library found

#]=======================================================================]

include(FindPackageHandleStandardArgs)

find_package(PkgConfig)
pkg_check_modules(PC_RMM QUIET RMM)

find_path(RMM_LOCATION  "rmm"
          HINTS
            "$ENV{RMM_ROOT}/include"
            "$ENV{CONDA_PREFIX}/include/rmm"
            "$ENV{CONDA_PREFIX}/include"
          PATHS
            ${PC_RMM_INCLUDE_DIRS}
          PATH_SUFFIXES
            include/rmm)

set(RMM_VERSION ${PC_RMM_VERSION})

mark_as_advanced(RMM_FOUND RMM_LOCATION RMM_VERSION)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RMM
    REQUIRED_VARS RMM_LOCATION
    VERSION_VAR RMM_VERSION
)

# list(APPEND RMM_NAMES RMM libRMM)
# set(_RMM_VERSION_SUFFIXES )

# foreach(v IN LISTS _RMM_VERSION_SUFFIXES)
#   list(APPEND RMM_NAMES RMM${v} libRMM${v})
#   list(APPEND RMM_NAMES RMM.${v} libRMM.${v})
# endforeach()
# unset(_RMM_VERSION_SUFFIXES)

# find_library(RMM_LIBRARY_RELEASE NAMES ${RMM_NAMES}
#              HINTS ${RMM_INSTALL_DIR}
#              PATH_SUFFIXES lib)

# include(${CMAKE_ROOT}/Modules/SelectLibraryConfigurations.cmake)
# select_library_configurations(RMM)
# mark_as_advanced(RMM_LIBRARY_RELEASE)
# unset(RMM_NAMES)

# Set by select_library_configurations(), but we want the one from
# find_package_handle_standard_args() below.
# unset(RMM_FOUND)

if (RMM_FOUND)
  set(RMM_INCLUDE_DIRS ${RMM_LOCATION})
  # set(RMM_LIBRARY ${RMM_LIBRARY})

  if(NOT TARGET RMM::RMM)
    add_library(RMM::RMM INTERFACE IMPORTED)
    set_target_properties(RMM::RMM PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${RMM_INCLUDE_DIRS}")
    # if(EXISTS "${RMM_LIBRARY}")
    #   set_target_properties(RMM::RMM PROPERTIES
    #     IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    #     IMPORTED_LOCATION "${RMM_LIBRARY}")
    # endif()
  endif()
endif ()


find_package_handle_standard_args(RMM
                                  REQUIRED_VARS RMM_LOCATION
                                  VERSION_VAR RMM_VERSION_STRING)

mark_as_advanced(RMM_LOCATION)


message(STATUS "RMM: RMM_INCLUDE_DIRS set to ${RMM_INCLUDE_DIRS}") 