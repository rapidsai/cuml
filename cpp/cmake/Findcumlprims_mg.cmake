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

#[=======================================================================[.rst:
Findcumlprims_mg
--------

Find libcumlprims_mg

Imported targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` target:

``CUMLPRIMS_MG::CUMLPRIMS_MG``
  The libcumlprims_mg library, if found.

Result variables
^^^^^^^^^^^^^^^^

This module will set the following variables in your project:

``CUMLPRIMS_MG_INCLUDE_DIRS``
  where to find cumlprims_mg.hpp , etc.
``CUMLPRIMS_MG_LIBRARIES``
  the libraries to link against to use libcumlprims_mg.
``CUMLPRIMS_MG_FOUND``
  If false, do not try to use CUMLPRIMS_MG.
``CUMLPRIMS_MG_VERSION_STRING``
  the version of the CUMLPRIMS_MG library found

#]=======================================================================]

find_path(CUMLPRIMS_LOCATION cumlprims.hpp
          HINTS ${CUMLPRIMS_MG_INSTALL_DIR}
          PATH_SUFFIXES include/cumlprims include)

list(APPEND CUMLPRIMS_MG_NAMES cumlprims libcumlprims)
set(_CUMLPRIMS_MG_VERSION_SUFFIXES 0)

foreach(v IN LISTS _CUMLPRIMS_MG_VERSION_SUFFIXES)
  list(APPEND CUMLPRIMS_MG_NAMES cumlprims${v} libcumlprims${v})
endforeach()
unset(_CUMLPRIMS_MG_VERSION_SUFFIXES)

find_library(CUMLPRIMS_MG_LIBRARY_RELEASE NAMES ${CUMLPRIMS_MG_NAMES}
             HINTS ${CUMLPRIMS_MG_INSTALL_DIR}
             PATH_SUFFIXES lib)

include(${CMAKE_ROOT}/Modules/SelectLibraryConfigurations.cmake)
select_library_configurations(CUMLPRIMS_MG)
mark_as_advanced(CUMLPRIMS_MG_LIBRARY_RELEASE)
unset(CUMLPRIMS_MG_NAMES)

# Set by select_library_configurations(), but we want the one from
# find_package_handle_standard_args() below.
unset(CUMLPRIMS_MG_FOUND)

if (CUMLPRIMS_MG_LIBRARY AND CUMLPRIMS_LOCATION)
  set(CUMLPRIMS_MG_INCLUDE_DIRS ${CUMLPRIMS_LOCATION} )
  set(CUMLPRIMS_MG_LIBRARY ${CUMLPRIMS_MG_LIBRARY})

  if(NOT TARGET CUMLPRIMS_MG::CUMLPRIMS_MG)
    add_library(CUMLPRIMS_MG::CUMLPRIMS_MG UNKNOWN IMPORTED)
    set_target_properties(CUMLPRIMS_MG::CUMLPRIMS_MG PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${CUMLPRIMS_MG_INCLUDE_DIRS}")
    if(EXISTS "${CUMLPRIMS_MG_LIBRARY}")
      set_target_properties(CUMLPRIMS_MG::CUMLPRIMS_MG PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
        IMPORTED_LOCATION "${CUMLPRIMS_MG_LIBRARY}")
    endif()
  endif()
endif ()


include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
find_package_handle_standard_args(CUMLPRIMS_MG
                                  REQUIRED_VARS CUMLPRIMS_MG_LIBRARY CUMLPRIMS_LOCATION
                                  VERSION_VAR CUMLPRIMS_MG_VERSION_STRING)

mark_as_advanced(CUMLPRIMS_LOCATION CUMLPRIMS_MG_LIBRARY)
