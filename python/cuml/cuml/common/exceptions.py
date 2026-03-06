#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from sklearn.exceptions import NotFittedError as _SklearnNotFittedError


class NotFittedError(_SklearnNotFittedError):
    """Exception class to raise if estimator is used before fitting.

    Inherits from sklearn's NotFittedError so that sklearn's estimator
    checks and except clauses catch it correctly.
    """
