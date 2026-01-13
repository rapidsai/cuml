# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Use CPM to find or clone CCCL
function(find_and_configure_cccl)
        include(${rapids-cmake-dir}/cpm/cccl.cmake)
        rapids_cpm_cccl()
endfunction()

find_and_configure_cccl()
