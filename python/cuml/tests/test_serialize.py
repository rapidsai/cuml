# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#

from distributed.protocol.serialize import serialize as ser
from cuml.naive_bayes.naive_bayes import MultinomialNB
import pickle
from cuml.internals.safe_imports import gpu_only_import

cp = gpu_only_import("cupy")
cupyx = gpu_only_import("cupyx")


def test_naive_bayes_cuda():
    """
    Assuming here that the Dask serializers are well-tested.
    This test-case is only validating that the Naive Bayes class
    actually gets registered w/ `dask` and `cuda` serializers.
    """

    mnb = MultinomialNB()

    X = cupyx.scipy.sparse.random(1, 5)
    y = cp.array([0])

    mnb.fit(X, y)

    # Unfortunately, Dask has no `unregister` function and Pytest
    # shares the same process so cannot test the base-state here.

    stype, sbytes = ser(mnb, serializers=["cuda"])
    assert stype["serializer"] == "cuda"

    stype, sbytes = ser(mnb, serializers=["dask"])
    assert stype["serializer"] == "dask"

    stype, sbytes = ser(mnb, serializers=["pickle"])
    assert stype["serializer"] == "pickle"


def test_cupy_sparse_patch():

    sp = cupyx.scipy.sparse.random(50, 2, format="csr")

    pickled = pickle.dumps(sp)

    sp_deser = pickle.loads(pickled)

    # Using internal API pieces only until
    # https://github.com/cupy/cupy/issues/3061
    # is fixed.
    assert sp_deser._descr.descriptor != sp._descr.descriptor
