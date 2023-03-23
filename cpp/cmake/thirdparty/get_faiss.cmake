#=============================================================================
# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
    set(oneValueArgs VERSION REPOSITORY PINNED_TAG BUILD_STATIC_LIBS EXCLUDE_FROM_ALL)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    if(CUML_USE_RAFT_NN)
        rapids_find_generate_module(faiss
                HEADER_NAMES  faiss/IndexFlat.h
                LIBRARY_NAMES faiss
                )

        set(BUILD_SHARED_LIBS ON)
        if (PKG_BUILD_STATIC_LIBS)
            set(BUILD_SHARED_LIBS OFF)
            set(CPM_DOWNLOAD_faiss ON)
        endif()

        rapids_cpm_find(faiss ${PKG_VERSION}
                GLOBAL_TARGETS     faiss::faiss
                CPM_ARGS
                GIT_REPOSITORY   ${PKG_REPOSITORY}
                GIT_TAG          ${PKG_PINNED_TAG}
                EXCLUDE_FROM_ALL ${PKG_EXCLUDE_FROM_ALL}
                OPTIONS
                "FAISS_ENABLE_PYTHON OFF"
                "CUDAToolkit_ROOT ${CUDAToolkit_LIBRARY_DIR}"
                "FAISS_ENABLE_GPU ON"
                "BUILD_TESTING OFF"
                "CMAKE_MESSAGE_LOG_LEVEL VERBOSE"
                "FAISS_USE_CUDA_TOOLKIT_STATIC ${CUDA_STATIC_RUNTIME}"
                )

        if(TARGET faiss AND NOT TARGET faiss::faiss)
            add_library(faiss::faiss ALIAS faiss)
        endif()

        if(faiss_ADDED)
            rapids_export(BUILD faiss
                    EXPORT_SET faiss-targets
                    GLOBAL_TARGETS faiss
                    NAMESPACE faiss::)
        endif()
    endif()

    # We generate the faiss-config files when we built faiss locally, so always do `find_dependency`
    rapids_export_package(BUILD OpenMP cuml-exports) # faiss uses openMP but doesn't export a need for it
    rapids_export_package(BUILD faiss cuml-exports GLOBAL_TARGETS faiss::faiss faiss)
    rapids_export_package(INSTALL faiss cuml-exports GLOBAL_TARGETS faiss::faiss faiss)

    # Tell cmake where it can find the generated faiss-config.cmake we wrote.
    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(BUILD faiss [=[${CMAKE_CURRENT_LIST_DIR}]=] cuml-exports)
endfunction()

if(NOT CUML_FAISS_GIT_TAG)
    # TODO: Remove this once faiss supports FAISS_USE_CUDA_TOOLKIT_STATIC
    # (https://github.com/facebookresearch/faiss/pull/2446)
    set(CUML_FAISS_GIT_TAG fea/statically-link-ctk-v1.7.0)
    # set(RAFT_FAISS_GIT_TAG bde7c0027191f29c9dadafe4f6e68ca0ee31fb30)
endif()

if(NOT CUML_FAISS_GIT_REPOSITORY)
    # TODO: Remove this once faiss supports FAISS_USE_CUDA_TOOLKIT_STATIC
    # (https://github.com/facebookresearch/faiss/pull/2446)
    set(CUML_FAISS_GIT_REPOSITORY https://github.com/trxcllnt/faiss.git)
    # set(RAFT_FAISS_GIT_REPOSITORY https://github.com/facebookresearch/faiss.git)
endif()

find_and_configure_faiss(VERSION    1.7.0
        REPOSITORY  ${CUML_FAISS_GIT_REPOSITORY}
        PINNED_TAG  ${CUML_FAISS_GIT_TAG}
        BUILD_STATIC_LIBS ${CUML_USE_FAISS_STATIC}
        EXCLUDE_FROM_ALL ${CUML_EXCLUDE_FAISS_FROM_ALL})