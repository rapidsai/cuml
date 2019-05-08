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

# Finds clang-tidy exe based on the PATH env variable
string(REPLACE ":" ";" EnvPath $ENV{PATH})
find_program(ClangTidy_EXE
  NAMES clang-tidy
  PATHS EnvPath
  DOC "path to clang-tidy exe")
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ClangTidy DEFAULT_MSG
  ClangTidy_EXE)

# TODO: add a clang_tidy dependency on the existing targets
