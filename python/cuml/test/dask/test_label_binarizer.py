# Copyright (c) 2019, NVIDIA CORPORATION.
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

import pytest
from cuml.dask.preprocessing import LabelBinarizer
from cuml.test.utils import array_equal
from dask.distributed import Client

import cudf
import dask_cudf

import numpy as np
import cupy as cp


@pytest.mark.parametrize(
    "labels", [([1, 4, 5, 2, 0, 1, 6, 2, 3, 4],
        [4, 2, 6, 3, 2, 0, 1]),
        ([9, 8, 2, 1, 3, 4],
        [8, 2, 1, 2, 2])]
)
def test_basic_functions(labels, cluster):

    client = Client(cluster)

    fit_labels, xform_labels = labels

    s = cudf.Series(np.array(fit_labels, dtype=np.int32))
    df = dask_cudf.from_cudf(s, npartitions=2)

    s2 = cudf.Series(np.array(xform_labels, dtype=np.int32))
    df2 = dask_cudf.from_cudf(s2, npartitions=2)

    binarizer = LabelBinarizer(client=client)
    binarizer.fit(df)

    assert array_equal(cp.asnumpy(binarizer.classes_),
                       np.unique(cp.asnumpy(fit_labels)))

    xformed = binarizer.transform(df2)

    assert xformed.compute().shape[1] == binarizer.classes_.shape[0]

    original = binarizer.inverse_transform(xformed)

    test = cp.asarray(original.compute().to_gpu_array())

    print("Test: " + str(test))

    assert array_equal(cp.asnumpy(test), xform_labels)

