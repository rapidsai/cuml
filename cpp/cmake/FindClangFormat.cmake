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

# Finds clang-format exe based on the PATH env variable
string(REPLACE ":" ";" EnvPath $ENV{PATH})
find_program(ClangFormat_EXE
  NAMES clang-format
  PATHS EnvPath
  DOC "path to clang-format exe")
find_program(ClangFormat_PY
  NAMES run-clang-format.py
  PATHS ${PROJECT_SOURCE_DIR}/scripts
  DOC "path to run-clang-format python script")

# Figure out the version of clang-format, if found
if(ClangFormat_EXE)
  execute_process(COMMAND ${ClangFormat_EXE} --version
    OUTPUT_VARIABLE __cf_version_out
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REGEX REPLACE
    "^clang-format version ([0-9.-]+).*$" "\\1"
    ClangFormat_VERSION_STRING
    "${__cf_version_out}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ClangFormat
  REQUIRED_VARS ClangFormat_EXE ClangFormat_PY
  VERSION_VAR ClangFormat_VERSION_STRING)

include(CMakeParseArguments)

set(ClangFormat_TARGET format)

# clang formatting as a target in the final build stage
function(add_clang_format)
  if(ClangFormat_FOUND)
    set(options "")
    set(oneValueArgs DSTDIR SRCDIR)
    set(multiValueArgs "")
    cmake_parse_arguments(cf "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    # to flag violations
    add_custom_target(${ClangFormat_TARGET}
      ALL
      COMMAND python
        ${ClangFormat_PY}
          -dstdir ${cf_DSTDIR}
          -exe ${ClangFormat_EXE}
          -onlyChangedFiles
      COMMENT "Run clang-format on the cpp source files"
      WORKING_DIRECTORY ${cf_SRCDIR})
    # to fix the flagged violations (only to be run locally!)
    add_custom_target(fix-${ClangFormat_TARGET}
      COMMAND python
      ${ClangFormat_PY}
        -dstdir ${cf_DSTDIR}
        -exe ${ClangFormat_EXE}
        -onlyChangedFiles
        -inplace
      COMMENT "Run the inplace fix for clang-format flagged violations"
      WORKING_DIRECTORY ${cf_SRCDIR})
  else()
    message("add_clang_format: clang-format exe not found")
  endif()
endfunction(add_clang_format)
