#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
def __getattr__(name):
    import warnings

    if name in ("set_global_output_type", "using_output_type"):
        warnings.warn(
            f"Accessing {name!r} from the `cuml.internals.memory_utils` "
            # rapids-pre-commit-hooks: disable-next-line[verify-hardcoded-version]
            f"namespace is deprecated and will be removed in 26.04. Please "
            f"use `cuml.{name}` instead.",
            FutureWarning,
        )
        import cuml.internals.outputs as mod

        return getattr(mod, name)
    raise AttributeError(
        f"module 'cuml.internals.memory_utils' has no attribute {name!r}"
    )
