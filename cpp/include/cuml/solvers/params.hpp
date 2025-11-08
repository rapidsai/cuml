/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace ML {

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

};  // namespace ML
