# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================


# todo: disable _mg for singlegpu
set(cython_sources
    base.pyx
    base_mg.pyx
    elastc_net.pyx
    linear_regression.pyx
    linear_regression_mg.pyx
    logistic_regression.pyx
    mbsgd_classifier.pyx
    mbsgd_regressor.pyx
    ridge.pyx
    ridge_mg.pyx
)

set(linked_libraries cuml::${CUML_CPP_TARGET})
rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}"
)

# TODO: Finding NumPy currently requires finding Development due to a bug in CMake. This bug was
# fixed in https://gitlab.kitware.com/cmake/cmake/-/merge_requests/7410 and will be available in
# CMake 3.24, so we can remove the Development component once we upgrade to CMake 3.24.
find_package(Python REQUIRED COMPONENTS Development NumPy)
set(targets_using_numpy
    base
    base_mg
    elastc_net
    linear_regression
    linear_regression_mg
    logistic_regression
    mbsgd_classifier
    mbsgd_regressor
    ridge
    ridge_mg
    )
foreach(target IN LISTS targets_using_numpy)
  target_include_directories(${target} PRIVATE "${Python_NumPy_INCLUDE_DIRS}")
endforeach()

foreach(cython_module IN LISTS RAPIDS_CYTHON_CREATED_TARGETS)
  set_target_properties(${cython_module} PROPERTIES INSTALL_RPATH "\$ORIGIN;\$ORIGIN/cpp")
endforeach()
