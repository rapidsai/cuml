# Copyright (c) 2020-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cupy as cp
import dask
import numpy as np
import pytest

from cuml.dask.preprocessing import LabelBinarizer
from cuml.testing.utils import array_equal


@pytest.mark.parametrize(
    "labels",
    [
        ([1, 4, 5, 2, 0, 1, 6, 2, 3, 4], [4, 2, 6, 3, 2, 0, 1]),
        ([9, 8, 2, 1, 3, 4], [8, 2, 1, 2, 2]),
    ],
)
@pytest.mark.parametrize("multipart", [True, False])
def test_basic_functions(labels, multipart, client):

    fit_labels, xform_labels = labels

    s = cp.asarray(fit_labels, dtype=np.int32)
    df = dask.array.from_array(s)

    s2 = cp.asarray(xform_labels, dtype=np.int32)
    df2 = dask.array.from_array(s2)

    if multipart:
        df = df.rechunk((1,))
        df2 = df2.rechunk((1,))

    binarizer = LabelBinarizer(client=client, sparse_output=False)
    binarizer.fit(df)

    assert array_equal(
        cp.asnumpy(binarizer.classes_), np.unique(cp.asnumpy(s))
    )

    xformed = binarizer.transform(df2)

    xformed = xformed.map_blocks(lambda x: x.get(), dtype=cp.float32)
    xformed.compute_chunk_sizes()

    assert xformed.compute().shape[1] == binarizer.classes_.shape[0]

    original = binarizer.inverse_transform(xformed)
    test = original.compute()

    assert array_equal(cp.asnumpy(test), xform_labels)


@pytest.mark.parametrize(
    "labels",
    [
        ([1, 4, 5, 2, 0, 1, 6, 2, 3, 4], [4, 2, 6, 3, 2, 0, 1]),
        ([9, 8, 2, 1, 3, 4], [8, 2, 1, 2, 2]),
    ],
)
@pytest.mark.xfail(
    raises=ValueError,
    reason="Sparse output disabled until "
    "Dask supports sparse CuPy "
    "arrays",
)
def test_sparse_output_fails(labels, client):

    LabelBinarizer(client=client, sparse_output=True)
