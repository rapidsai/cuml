/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/export.hpp>
namespace CUML_EXPORT ML {

enum lr_type {
  OPTIMAL,
  CONSTANT,
  INVSCALING,
  ADAPTIVE,
};

enum loss_funct {
  SQRD_LOSS,
  HINGE,
  LOG,
};

enum penalty { NONE, L1, L2, ELASTICNET };

};  // namespace CUML_EXPORT ML
