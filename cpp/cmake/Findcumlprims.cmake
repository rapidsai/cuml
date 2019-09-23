## Copyright (c) 2019, NVIDIA CORPORATION.
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
FindCUMLPRIMS
--------

Find libcumlprims, the NVIDIA Collective Communication Library. A hint to find CUMLPRIMS
can be provided by setting CUMLPRIMS_INSTALL_DIR.

Imported targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` target:

``CUMLPRIMS::CUMLPRIMS``
  The libcumlprims library, if found.

Result variables
^^^^^^^^^^^^^^^^

This module will set the following variables in your project:

``CUMLPRIMS_INCLUDE_DIRS``
  where to find cumlprims.h , etc.
``CUMLPRIMS_LIBRARIES``
  the libraries to link against to use CUMLPRIMS.
``CUMLPRIMS_FOUND``
  If false, do not try to use CUMLPRIMS.
``CUMLPRIMS_VERSION_STRING``
  the version of the CUMLPRIMS library found

#]=======================================================================]

find_path(CUMLPRIMS_CUMLPRIMS_INCLUDE_DIR cumlprims.hpp HINTS ${CUMLPRIMS_INSTALL_DIR} PATH_SUFFIXES include/cumlprims)

list(APPEND CUMLPRIMS_NAMES cumlprims libcumlprims)

# For compatibility with versions prior to this multi-config search, honor
# any CUMLPRIMS_LIBRARY that is already specified and skip the search.
if(NOT CUMLPRIMS_LIBRARY)
  find_library(CUMLPRIMS_LIBRARY_RELEASE NAMES ${CUMLPRIMS_NAMES} HINTS ${CUMLPRIMS_INSTALL_DIR} PATH_SUFFIXES lib)
  include(${CMAKE_ROOT}/Modules/SelectLibraryConfigurations.cmake)
  select_library_configurations(CUMLPRIMS)
  mark_as_advanced(CUMLPRIMS_LIBRARY_RELEASE)
endif()
unset(CUMLPRIMS_NAMES)

# Set by select_library_configurations(), but we want the one from
# find_package_handle_standard_args() below.
unset(CUMLPRIMS_FOUND)

if (CUMLPRIMS_LIBRARY AND CUMLPRIMS_CUMLPRIMS_INCLUDE_DIR)
  set(CUMLPRIMS_INCLUDE_DIRS ${CUMLPRIMS_CUMLPRIMS_INCLUDE_DIR} )
  set(CUMLPRIMS_LIBRARY ${CUMLPRIMS_LIBRARY})

  if(NOT TARGET CUMLPRIMS::CUMLPRIMS)
    add_library(CUMLPRIMS::CUMLPRIMS UNKNOWN IMPORTED)
    set_target_properties(CUMLPRIMS::CUMLPRIMS PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${CUMLPRIMS_INCLUDE_DIRS}")
    if(EXISTS "${CUMLPRIMS_LIBRARY}")
      set_target_properties(CUMLPRIMS::CUMLPRIMS PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION "${CUMLPRIMS_LIBRARY}")
    endif()
  endif()
endif ()

if (CUMLPRIMS_CUMLPRIMS_INCLUDE_DIR AND EXISTS "${CUMLPRIMS_CUMLPRIMS_INCLUDE_DIR}/cumlprims.h")
  file(STRINGS "${CUMLPRIMS_CUMLPRIMS_INCLUDE_DIR}/cumlprims.h" cumlprims_major_version_str REGEX "^#define[ \t]+CUMLPRIMS_MAJOR[ \t]+[0-9]+")
  string(REGEX REPLACE "^#define[ \t]+CUMLPRIMS_MAJOR[ \t]+([0-9]+)" "\\1" cumlprims_major_version_str "${cumlprims_major_version_str}")

  file(STRINGS "${CUMLPRIMS_CUMLPRIMS_INCLUDE_DIR}/cumlprims.h" cumlprims_minor_version_str REGEX "^#define[ \t]+CUMLPRIMS_MINOR[ \t]+[0-9]+")
  string(REGEX REPLACE "^#define[ \t]+CUMLPRIMS_MINOR[ \t]+([0-9]+)" "\\1" cumlprims_minor_version_str "${cumlprims_minor_version_str}")

  file(STRINGS "${CUMLPRIMS_CUMLPRIMS_INCLUDE_DIR}/cumlprims.h" cumlprims_patch_version_str REGEX "^#define[ \t]+CUMLPRIMS_PATCH[ \t]+[0-9]+")
  string(REGEX REPLACE "^#define[ \t]+CUMLPRIMS_PATCH[ \t]+([0-9]+)" "\\1" cumlprims_patch_version_str "${cumlprims_patch_version_str}")

  file(STRINGS "${CUMLPRIMS_CUMLPRIMS_INCLUDE_DIR}/cumlprims.h" cumlprims_suffix_version_str REGEX "^#define[ \t]+CUMLPRIMS_SUFFIX[ \t]+\".*\"")
  string(REGEX REPLACE "^#define[ \t]+CUMLPRIMS_SUFFIX[ \t]+\"(.*)\"" "\\1" cumlprims_suffix_version_str "${cumlprims_suffix_version_str}")

  set(CUMLPRIMS_VERSION_STRING "${cumlprims_major_version_str}.${cumlprims_minor_version_str}.${cumlprims_patch_version_str}${cumlprims_suffix_version_str}")

  unset(cumlprims_major_version_str)
  unset(cumlprims_minor_version_str)
  unset(cumlprims_patch_version_str)
  unset(cumlprims_suffix_version_str)
endif ()

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
find_package_handle_standard_args(CUMLPRIMS
                                  REQUIRED_VARS CUMLPRIMS_LIBRARY CUMLPRIMS_CUMLPRIMS_INCLUDE_DIR
                                  VERSION_VAR CUMLPRIMS_VERSION_STRING)

mark_as_advanced(CUMLPRIMS_CUMLPRIMS_INCLUDE_DIR CUMLPRIMS_LIBRARY)
