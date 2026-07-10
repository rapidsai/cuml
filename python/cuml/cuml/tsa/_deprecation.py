# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import warnings


def warn_deprecated_tsa_api(api_name):
    warnings.warn(
        (
            f"`{api_name}` was deprecated in version 26.08 and will be "
            "removed in version 26.12. `cuml.tsa` was "
            "deprecated in version 26.08 and will be removed in version 26.12."
        ),
        FutureWarning,
        stacklevel=2,
    )
