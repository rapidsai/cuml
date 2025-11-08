#
# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """
