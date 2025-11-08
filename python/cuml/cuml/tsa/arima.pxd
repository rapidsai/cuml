#
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

cdef extern from "cuml/tsa/arima_common.h" namespace "ML" nogil:
    ctypedef struct ARIMAOrder:
        int p       # Basic order
        int d
        int q
        int P       # Seasonal order
        int D
        int Q
        int s       # Seasonal period
        int k       # Fit intercept?
        int n_exog  # Number of exogenous regressors
