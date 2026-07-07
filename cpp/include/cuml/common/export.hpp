/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#if (defined(__GNUC__) && !defined(__MINGW32__) && !defined(__MINGW64__))
#define CUML_EXPORT __attribute__((visibility("default")))
#define CUML_HIDDEN __attribute__((visibility("hidden")))
#else
#define CUML_EXPORT
#define CUML_HIDDEN
#endif
