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
FindFAISS
--------

Find FAISS libraries (libfaiss, libucm, libuct, libucs) for the FAISS Point-to-Point Protocol Library.
Hints to find FAISS can be provided by setting FAISS_INSTALL_DIR and FAISS_INCLUDE_DIR.

Imported targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` target:

``FAISS::FAISS``
  The FAISS libraries, if found.

Result variables
^^^^^^^^^^^^^^^^

This module will set the following variables in your project:

``FAISS_INCLUDE_DIRS``
  where to find IndexFlat.h , etc.
``FAISS_LIBRARIES``
  the libraries to link against to use FAISS.
``FAISS_FOUND``
  If false, do not try to use FAISS.
``FAISS_VERSION_STRING``
  the version of the FAISS library found

#]=======================================================================]

find_path(FAISS_FAISS_INCLUDE_DIR NAMES faiss/IndexFlat.h HINTS ${FAISS_INSTALL_DIR} PATH_SUFFIXES include)

list(APPEND FAISS_NAMES faiss libfaiss)
set(_FAISS_VERSION_SUFFIXES 0)

foreach(v IN LISTS _FAISS_VERSION_SUFFIXES)
  list(APPEND FAISS_NAMES faiss${v} libfaiss${v})
endforeach()
unset(_FAISS_VERSION_SUFFIXES)
# For compatibility with versions prior to this multi-config search, honor
# any FAISS_LIBRARY that is already specified and skip the search.
if(NOT FAISS_LIBRARY)
  find_library(FAISS_LIBRARY_RELEASE NAMES ${FAISS_NAMES} HINTS ${FAISS_INSTALL_DIR} PATH_SUFFIXES lib)
  include(${CMAKE_ROOT}/Modules/SelectLibraryConfigurations.cmake)
  select_library_configurations(FAISS)
  mark_as_advanced(FAISS_LIBRARY_RELEASE)
endif()
unset(FAISS_NAMES)

# Set by select_library_configurations(), but we want the one from
# find_package_handle_standard_args() below.
unset(FAISS_FOUND)

if (FAISS_LIBRARY AND FAISS_FAISS_INCLUDE_DIR)
  set(FAISS_INCLUDE_DIRS ${FAISS_FAISS_INCLUDE_DIR} )
  set(FAISS_LIBRARY ${FAISS_LIBRARY})

  if(NOT TARGET FAISS::FAISS)
    add_library(FAISS::FAISS UNKNOWN IMPORTED)
    set_target_properties(FAISS::FAISS PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${FAISS_INCLUDE_DIRS}")
    if(EXISTS "${FAISS_LIBRARY}")
      set_target_properties(FAISS::FAISS PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION "${FAISS_LIBRARY}")
    endif()
  endif()
endif ()

if (FAISS_FAISS_INCLUDE_DIR AND EXISTS "${FAISS_FAISS_INCLUDE_DIR}/IndexFlat.h")
  file(STRINGS "${FAISS_FAISS_INCLUDE_DIR}/IndexFlat.h" faiss_major_version_str REGEX "^#define[ \t]+FAISS_MAJOR[ \t]+[0-9]+")
  string(REGEX REPLACE "^#define[ \t]+FAISS_MAJOR[ \t]+([0-9]+)" "\\1" faiss_major_version_str "${faiss_major_version_str}")

  file(STRINGS "${FAISS_FAISS_INCLUDE_DIR}/IndexFlat.h" faiss_minor_version_str REGEX "^#define[ \t]+FAISS_MINOR[ \t]+[0-9]+")
  string(REGEX REPLACE "^#define[ \t]+FAISS_MINOR[ \t]+([0-9]+)" "\\1" faiss_minor_version_str "${faiss_minor_version_str}")

  file(STRINGS "${FAISS_FAISS_INCLUDE_DIR}/IndexFlat.h" faiss_patch_version_str REGEX "^#define[ \t]+FAISS_PATCH[ \t]+[0-9]+")
  string(REGEX REPLACE "^#define[ \t]+FAISS_PATCH[ \t]+([0-9]+)" "\\1" faiss_patch_version_str "${faiss_patch_version_str}")

  file(STRINGS "${FAISS_FAISS_INCLUDE_DIR}/IndexFlat.h" faiss_suffix_version_str REGEX "^#define[ \t]+FAISS_SUFFIX[ \t]+\".*\"")
  string(REGEX REPLACE "^#define[ \t]+FAISS_SUFFIX[ \t]+\"(.*)\"" "\\1" faiss_suffix_version_str "${faiss_suffix_version_str}")

  set(FAISS_VERSION_STRING "${faiss_major_version_str}.${faiss_minor_version_str}.${faiss_patch_version_str}${faiss_suffix_version_str}")

  unset(faiss_major_version_str)
  unset(faiss_minor_version_str)
  unset(faiss_patch_version_str)
  unset(faiss_suffix_version_str)
endif ()

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
find_package_handle_standard_args(FAISS
                                  REQUIRED_VARS FAISS_LIBRARY FAISS_FAISS_INCLUDE_DIR
                                  VERSION_VAR FAISS_VERSION_STRING)

mark_as_advanced(FAISS_FAISS_INCLUDE_DIR FAISS_LIBRARY)