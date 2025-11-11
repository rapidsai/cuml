# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pytest
import scipy.sparse
from sklearn.preprocessing import LabelBinarizer as skLB

from cuml.preprocessing import LabelBinarizer
from cuml.testing.utils import array_equal


@pytest.mark.parametrize(
    "labels",
    [
        ([1, 4, 5, 2, 0, 1, 6, 2, 3, 4], [4, 2, 6, 3, 2, 0, 1]),
        ([9, 8, 2, 1, 3, 4], [8, 2, 1, 2, 2]),
    ],
)
@pytest.mark.parametrize("dtype", [cp.int32, cp.int64])
@pytest.mark.parametrize("sparse_output", [True, False])
def test_basic_functions(labels, dtype, sparse_output):

    fit_labels, xform_labels = labels

    skl_bin = skLB(sparse_output=sparse_output)
    skl_bin.fit(fit_labels)

    fit_labels = cp.asarray(fit_labels, dtype=dtype)
    xform_labels = cp.asarray(xform_labels, dtype=dtype)

    binarizer = LabelBinarizer(sparse_output=sparse_output)
    binarizer.fit(fit_labels)

    assert array_equal(binarizer.classes_.get(), np.unique(fit_labels.get()))

    xformed = binarizer.transform(xform_labels)

    if sparse_output:
        skl_bin_xformed = skl_bin.transform(xform_labels.get())

        skl_csr = scipy.sparse.coo_matrix(skl_bin_xformed).tocsr()
        cuml_csr = xformed

        array_equal(skl_csr.data, cuml_csr.data.get())

        # #todo: Support sparse inputs
        # xformed = xformed.todense().astype(dtype)

    assert xformed.shape[1] == binarizer.classes_.shape[0]

    original = binarizer.inverse_transform(xformed)

    assert array_equal(original.get(), xform_labels.get())
