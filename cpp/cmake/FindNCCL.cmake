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
FindNCCL
--------

Find libnccl, the NVIDIA Collective Communication Library. A hint to find NCCL
can be provided by setting NCCL_INSTALL_DIR.

Imported targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` target:

``NCCL::NCCL``
  The libnccl library, if found.

Result variables
^^^^^^^^^^^^^^^^

This module will set the following variables in your project:

``NCCL_INCLUDE_DIRS``
  where to find nccl.h , etc.
``NCCL_LIBRARIES``
  the libraries to link against to use NCCL.
``NCCL_FOUND``
  If false, do not try to use NCCL.
``NCCL_VERSION_STRING``
  the version of the NCCL library found

#]=======================================================================]

find_path(NCCL_NCCL_INCLUDE_DIR nccl.h HINTS ${NCCL_INSTALL_DIR} PATH_SUFFIXES include)

#TODO: Does this need to support finding the static library?

list(APPEND NCCL_NAMES nccl libnccl)
set(_NCCL_VERSION_SUFFIXES 2)

foreach(v IN LISTS _NCCL_VERSION_SUFFIXES)
  list(APPEND NCCL_NAMES nccl${v} libnccl${v})
endforeach()
unset(_NCCL_VERSION_SUFFIXES)
# For compatibility with versions prior to this multi-config search, honor
# any NCCL_LIBRARY that is already specified and skip the search.
if(NOT NCCL_LIBRARY)
  find_library(NCCL_LIBRARY_RELEASE NAMES ${NCCL_NAMES} HINTS ${NCCL_INSTALL_DIR} PATH_SUFFIXES lib)
  include(${CMAKE_ROOT}/Modules/SelectLibraryConfigurations.cmake)
  select_library_configurations(NCCL)
  mark_as_advanced(NCCL_LIBRARY_RELEASE)
endif()
unset(NCCL_NAMES)

# Set by select_library_configurations(), but we want the one from
# find_package_handle_standard_args() below.
unset(NCCL_FOUND)

if (NCCL_LIBRARY AND NCCL_NCCL_INCLUDE_DIR)
  set(NCCL_INCLUDE_DIRS ${NCCL_NCCL_INCLUDE_DIR} )
  set(NCCL_LIBRARY ${NCCL_LIBRARY})

  if(NOT TARGET NCCL::NCCL)
    add_library(NCCL::NCCL UNKNOWN IMPORTED)
    set_target_properties(NCCL::NCCL PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIRS}")
    if(EXISTS "${NCCL_LIBRARY}")
      set_target_properties(NCCL::NCCL PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION "${NCCL_LIBRARY}")
    endif()
  endif()
endif ()

if (NCCL_NCCL_INCLUDE_DIR AND EXISTS "${NCCL_NCCL_INCLUDE_DIR}/nccl.h")
  file(STRINGS "${NCCL_NCCL_INCLUDE_DIR}/nccl.h" nccl_major_version_str REGEX "^#define[ \t]+NCCL_MAJOR[ \t]+[0-9]+")
  string(REGEX REPLACE "^#define[ \t]+NCCL_MAJOR[ \t]+([0-9]+)" "\\1" nccl_major_version_str "${nccl_major_version_str}")

  file(STRINGS "${NCCL_NCCL_INCLUDE_DIR}/nccl.h" nccl_minor_version_str REGEX "^#define[ \t]+NCCL_MINOR[ \t]+[0-9]+")
  string(REGEX REPLACE "^#define[ \t]+NCCL_MINOR[ \t]+([0-9]+)" "\\1" nccl_minor_version_str "${nccl_minor_version_str}")

  file(STRINGS "${NCCL_NCCL_INCLUDE_DIR}/nccl.h" nccl_patch_version_str REGEX "^#define[ \t]+NCCL_PATCH[ \t]+[0-9]+")
  string(REGEX REPLACE "^#define[ \t]+NCCL_PATCH[ \t]+([0-9]+)" "\\1" nccl_patch_version_str "${nccl_patch_version_str}")

  file(STRINGS "${NCCL_NCCL_INCLUDE_DIR}/nccl.h" nccl_suffix_version_str REGEX "^#define[ \t]+NCCL_SUFFIX[ \t]+\".*\"")
  string(REGEX REPLACE "^#define[ \t]+NCCL_SUFFIX[ \t]+\"(.*)\"" "\\1" nccl_suffix_version_str "${nccl_suffix_version_str}")

  set(NCCL_VERSION_STRING "${nccl_major_version_str}.${nccl_minor_version_str}.${nccl_patch_version_str}${nccl_suffix_version_str}")

  unset(nccl_major_version_str)
  unset(nccl_minor_version_str)
  unset(nccl_patch_version_str)
  unset(nccl_suffix_version_str)
endif ()

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
find_package_handle_standard_args(NCCL
                                  REQUIRED_VARS NCCL_LIBRARY NCCL_NCCL_INCLUDE_DIR
                                  VERSION_VAR NCCL_VERSION_STRING)

mark_as_advanced(NCCL_NCCL_INCLUDE_DIR NCCL_LIBRARY)
