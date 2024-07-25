# Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

find_package(Doxygen 1.9.1)

function(add_doxygen_target)
  if(Doxygen_FOUND)
    set(options "")
    set(oneValueArgs IN_DOXYFILE OUT_DOXYFILE CWD)
    set(multiValueArgs "")
    cmake_parse_arguments(dox "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    configure_file(${dox_IN_DOXYFILE} ${dox_OUT_DOXYFILE} @ONLY)
    add_custom_target(docs_cuml
      ${CMAKE_COMMAND} -E env "RAPIDS_VERSION=${RAPIDS_VERSION}" "RAPIDS_VERSION_MAJOR_MINOR=${RAPIDS_VERSION_MAJOR_MINOR}"
      ${DOXYGEN_EXECUTABLE} ${dox_OUT_DOXYFILE}
      WORKING_DIRECTORY ${dox_CWD}
      VERBATIM
      COMMENT "Generate doxygen docs")
  else()
    message("add_doxygen_target: doxygen exe not found")
  endif()
endfunction(add_doxygen_target)
