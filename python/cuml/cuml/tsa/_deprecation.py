# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import warnings

_REMOVAL_VERSION = "26.12"


def warn_deprecated_tsa_api(api_name):
    warnings.warn(
        (
            f"`{api_name}` was deprecated in version 26.08 and will be "
            f"removed in version {_REMOVAL_VERSION}. `cuml.tsa` was "
            "deprecated in version 26.08 and will be removed in version "
            f"{_REMOVAL_VERSION}."
        ),
        FutureWarning,
        stacklevel=2,
    )
