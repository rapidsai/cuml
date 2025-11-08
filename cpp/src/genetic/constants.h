/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/** @file constants.h Common GPU functionality + constants for all operations */

#pragma once

namespace cuml {
namespace genetic {

// Max number of threads per block to use with tournament and evaluation kernels
const int GENE_TPB = 256;

// Max size of stack used for AST evaluation
const int MAX_STACK_SIZE = 20;

}  // namespace genetic
}  // namespace cuml
