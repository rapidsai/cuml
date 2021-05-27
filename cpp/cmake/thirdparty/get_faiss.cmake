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

function(find_and_configure_faiss)
    set(oneValueArgs VERSION PINNED_TAG)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    rapids_find_generate_module(FAISS
        HEADER_NAMES  faiss/IndexFlat.h
        LIBRARY_NAMES faiss
    )

    rapids_cpm_find(FAISS ${PKG_VERSION}
        GLOBAL_TARGETS  faiss
        CPM_ARGS
          GIT_REPOSITORY  https://github.com/facebookresearch/faiss.git
          GIT_TAG         ${PKG_PINNED_TAG}
          OPTIONS
            "FAISS_ENABLE_PYTHON OFF"
            "BUILD_SHARED_LIBS OFF"
            "CUDAToolkit_ROOT ${CUDAToolkit_LIBRARY_DIR}"
            "FAISS_ENABLE_GPU ON"
            "BUILD_TESTING OFF"
            "CMAKE_MESSAGE_LOG_LEVEL VERBOSE"
    )

    if(FAISS_ADDED)
      set(FAISS_GPU_HEADERS ${FAISS_SOURCE_DIR} PARENT_SCOPE)
      add_library(FAISS::FAISS ALIAS faiss)
    endif()

endfunction()

find_and_configure_faiss(VERSION    1.7.0
                         PINNED_TAG  bde7c0027191f29c9dadafe4f6e68ca0ee31fb30
                        )
