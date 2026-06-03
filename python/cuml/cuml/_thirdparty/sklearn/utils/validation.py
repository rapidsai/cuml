# SPDX-FileCopyrightText: Olivier Grisel
# SPDX-FileCopyrightText: Gael Varoquaux
# SPDX-FileCopyrightText: Andreas Mueller
# SPDX-FileCopyrightText: Lars Buitinck
# SPDX-FileCopyrightText: Alexandre Gramfort
# SPDX-FileCopyrightText: Nicolas Tresegnie
# SPDX-FileCopyrightText: Sylvain Marie
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

# Original authors from Sckit-Learn:
#          Olivier Grisel
#          Gael Varoquaux
#          Andreas Mueller
#          Lars Buitinck
#          Alexandre Gramfort
#          Nicolas Tresegnie
#          Sylvain Marie
# License: BSD 3 clause


# This code originates from the Scikit-Learn library,
# it was since modified to allow GPU acceleration.
# This code is under BSD 3 clause license.
# Authors mentioned above do not endorse or promote this production.
import numbers

import cupy as cp
import cupyx.scipy.sparse as sp
import numpy as np


FLOAT_DTYPES = (np.float64, np.float32, np.float16)


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def _allclose_dense_sparse(x, y, rtol=1e-7, atol=1e-9):
    """Check allclose for sparse and dense data.

    Both x and y need to be either sparse or dense, they
    can't be mixed.

    Parameters
    ----------
    x : array-like or sparse matrix
        First array to compare.

    y : array-like or sparse matrix
        Second array to compare.

    rtol : float, optional
        relative tolerance; see numpy.allclose

    atol : float, optional
        absolute tolerance; see numpy.allclose. Note that the default here is
        more tolerant than the default for numpy.testing.assert_allclose, where
        atol=0.
    """
    if sp.issparse(x) and sp.issparse(y):
        x = x.tocsr()
        y = y.tocsr()
        x.sum_duplicates()
        y.sum_duplicates()
        return (cp.array_equal(x.indices, y.indices) and
                cp.array_equal(x.indptr, y.indptr) and
                cp.allclose(x.data, y.data, rtol=rtol, atol=atol))
    elif not sp.issparse(x) and not sp.issparse(y):
        return cp.allclose(x, y, rtol=rtol, atol=atol)
    raise ValueError("Can only compare two sparse matrices, not a sparse "
                     "matrix and an array")
