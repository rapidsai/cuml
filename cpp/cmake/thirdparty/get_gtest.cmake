#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
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
#=============================================================================

function(find_and_configure_gtest VERSION)

    if(TARGET GTest::gtest)
        return()
    endif()

    rapids_cpm_find(GTest ${VERSION}
        GLOBAL_TARGETS  gmock gmock_main gtest gtest_main GTest::gmock GTest::gtest GTest::gtest_main
        CPM_ARGS
            GIT_REPOSITORY  https://github.com/google/googletest.git
            GIT_TAG         release-${VERSION}
            GIT_SHALLOW     TRUE
            OPTIONS
                "INSTALL_GTEST ON"
            # googletest >= 1.10.0 provides a cmake config file -- use it if it exists
            FIND_PACKAGE_ARGUMENTS "CONFIG"
    )

    if(NOT TARGET GTest::gtest)
        add_library(GTest::gtest ALIAS gtest)
        add_library(GTest::gtest_main ALIAS gtest_main)
    endif()

endfunction()

find_and_configure_gtest(1.10.0)
