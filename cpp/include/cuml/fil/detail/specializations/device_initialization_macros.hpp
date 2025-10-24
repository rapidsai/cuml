/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuml/fil/detail/raft_proto/device_id.hpp>
#include <cuml/fil/detail/raft_proto/device_type.hpp>
#include <cuml/fil/detail/specializations/forest_macros.hpp>
/* Declare device initialization function for the types specified by the given
 * variant index */
#define CUML_FIL_INITIALIZE_DEVICE(template_type, variant_index)                     \
  template_type void                                                                 \
    initialize_device<CUML_FIL_FOREST(variant_index), raft_proto::device_type::gpu>( \
      raft_proto::device_id<raft_proto::device_type::gpu>);
