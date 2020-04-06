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

# Based on FindPNG.cmake from cmake 3.14.3

#[=======================================================================[.rst:
FindUCX
--------

Find UCX libraries (libucp, libucm, libuct, libucs) for the UCX Point-to-Point Protocol Library. 
Hints to find UCX can be provided by setting UCX_INSTALL_DIR and UCX_INCLUDE_DIR.

Imported targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` target:

``UCX::UCX``
  The UCX libraries, if found.

Result variables
^^^^^^^^^^^^^^^^

This module will set the following variables in your project:

``UCX_INCLUDE_DIRS``
  where to find ucp.h , etc.
``UCX_LIBRARIES``
  the libraries to link against to use UCX.
``UCX_FOUND``
  If false, do not try to use UCX.
``UCX_VERSION_STRING``
  the version of the UCX library found

#]=======================================================================]

find_path(UCX_UCX_INCLUDE_DIR NAMES ucp/api/ucp.h HINTS ${UCX_INSTALL_DIR} PATH_SUFFIXES include)

list(APPEND UCX_NAMES ucp libucp ucs libucs ucm libucm uct libuct)
set(_UCX_VERSION_SUFFIXES 0)

foreach(v IN LISTS _UCX_VERSION_SUFFIXES)
  list(APPEND UCX_NAMES ucp${v} libucp${v} ucs${v} libucs${v} ucm${v} libucm${v} uct${v} libuct${v} )
endforeach()
unset(_UCX_VERSION_SUFFIXES)
# For compatibility with versions prior to this multi-config search, honor
# any UCX_LIBRARY that is already specified and skip the search.
if(NOT UCX_LIBRARY)
  find_library(UCX_LIBRARY_RELEASE NAMES ${UCX_NAMES} HINTS ${UCX_INSTALL_DIR} PATH_SUFFIXES lib)
  include(${CMAKE_ROOT}/Modules/SelectLibraryConfigurations.cmake)
  select_library_configurations(UCX)
  mark_as_advanced(UCX_LIBRARY_RELEASE)
endif()
unset(UCX_NAMES)

# Set by select_library_configurations(), but we want the one from
# find_package_handle_standard_args() below.
unset(UCX_FOUND)

if (UCX_LIBRARY AND UCX_UCX_INCLUDE_DIR)
  set(UCX_INCLUDE_DIRS ${UCX_UCX_INCLUDE_DIR} )
  set(UCX_LIBRARY ${UCX_LIBRARY})

  if(NOT TARGET UCX::UCX)
    add_library(UCX::UCX UNKNOWN IMPORTED)
    set_target_properties(UCX::UCX PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${UCX_INCLUDE_DIRS}")
    if(EXISTS "${UCX_LIBRARY}")
      set_target_properties(UCX::UCX PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION "${UCX_LIBRARY}")
    endif()
  endif()
endif ()

if (UCX_UCX_INCLUDE_DIR AND EXISTS "${UCX_UCX_INCLUDE_DIR}/ucp.h")
  file(STRINGS "${UCX_UCX_INCLUDE_DIR}/ucp.h" ucx_major_version_str REGEX "^#define[ \t]+UCX_MAJOR[ \t]+[0-9]+")
  string(REGEX REPLACE "^#define[ \t]+UCX_MAJOR[ \t]+([0-9]+)" "\\1" ucx_major_version_str "${ucx_major_version_str}")

  file(STRINGS "${UCX_UCX_INCLUDE_DIR}/ucp.h" ucx_minor_version_str REGEX "^#define[ \t]+UCX_MINOR[ \t]+[0-9]+")
  string(REGEX REPLACE "^#define[ \t]+UCX_MINOR[ \t]+([0-9]+)" "\\1" ucx_minor_version_str "${ucx_minor_version_str}")

  file(STRINGS "${UCX_UCX_INCLUDE_DIR}/ucp.h" ucx_patch_version_str REGEX "^#define[ \t]+UCX_PATCH[ \t]+[0-9]+")
  string(REGEX REPLACE "^#define[ \t]+UCX_PATCH[ \t]+([0-9]+)" "\\1" ucx_patch_version_str "${ucx_patch_version_str}")

  file(STRINGS "${UCX_UCX_INCLUDE_DIR}/ucp.h" ucx_suffix_version_str REGEX "^#define[ \t]+UCX_SUFFIX[ \t]+\".*\"")
  string(REGEX REPLACE "^#define[ \t]+UCX_SUFFIX[ \t]+\"(.*)\"" "\\1" ucx_suffix_version_str "${ucx_suffix_version_str}")

  set(UCX_VERSION_STRING "${ucx_major_version_str}.${ucx_minor_version_str}.${ucx_patch_version_str}${ucx_suffix_version_str}")

  unset(ucx_major_version_str)
  unset(ucx_minor_version_str)
  unset(ucx_patch_version_str)
  unset(ucx_suffix_version_str)
endif ()

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
find_package_handle_standard_args(UCX
                                  REQUIRED_VARS UCX_LIBRARY UCX_UCX_INCLUDE_DIR
                                  VERSION_VAR UCX_VERSION_STRING)

mark_as_advanced(UCX_UCX_INCLUDE_DIR UCX_LIBRARY)
