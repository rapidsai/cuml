#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================

function(find_and_configure_rmm)
    include(${rapids-cmake-dir}/cpm/rmm.cmake)
    rapids_cpm_rmm(BUILD_EXPORT_SET cuml-exports
                   INSTALL_EXPORT_SET cuml-exports)
endfunction()

find_and_configure_rmm()
