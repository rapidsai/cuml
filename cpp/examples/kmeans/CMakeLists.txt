#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================

add_executable(kmeans_example kmeans_example.cpp)
target_link_libraries(kmeans_example cuml++ cuvs::cuvs)
