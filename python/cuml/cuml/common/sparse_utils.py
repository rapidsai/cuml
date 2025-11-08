#
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupyx
import scipy.sparse


def is_sparse(X):
    """
    Return true if X is sparse, false otherwise.
    Parameters
    ----------
    X : array-like, sparse-matrix

    Returns
    -------

    is_sparse : boolean
        is the input sparse?
    """
    return scipy.sparse.issparse(X) or cupyx.scipy.sparse.issparse(X)


def is_dense(X):
    return not is_sparse(X)
