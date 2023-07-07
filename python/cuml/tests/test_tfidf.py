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

from sklearn.feature_extraction.text import TfidfTransformer as SkTfidfTransfo
from cuml.feature_extraction.text import TfidfTransformer
from cuml.internals.safe_imports import gpu_only_import
import pytest
from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import("numpy")
cp = gpu_only_import("cupy")
cupyx = gpu_only_import("cupyx")


# data_ids correspond to data, order is important
data_ids = ["base_case", "diag", "empty_feature", "123", "empty_doc"]
data = [
    np.array(
        [
            [0, 1, 1, 1, 0, 0, 1, 0, 1],
            [0, 2, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 1, 1, 0, 1, 1, 1],
            [0, 1, 1, 1, 0, 0, 1, 0, 1],
        ]
    ),
    np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]]),
    np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0]]),
    np.array([[1], [2], [3]]),
    np.array([[1, 1, 1], [1, 1, 0], [0, 0, 0]]),
]


@pytest.mark.parametrize("data", data, ids=data_ids)
@pytest.mark.parametrize("norm", ["l1", "l2", None])
@pytest.mark.parametrize("use_idf", [True, False])
@pytest.mark.parametrize("smooth_idf", [True, False])
@pytest.mark.parametrize("sublinear_tf", [True, False])
@pytest.mark.filterwarnings(
    "ignore:divide by zero(.*):RuntimeWarning:" "sklearn[.*]"
)
def test_tfidf_transformer(data, norm, use_idf, smooth_idf, sublinear_tf):
    data_gpu = cp.array(data)

    tfidf = TfidfTransformer(
        norm=norm,
        use_idf=use_idf,
        smooth_idf=smooth_idf,
        sublinear_tf=sublinear_tf,
    )
    sk_tfidf = SkTfidfTransfo(
        norm=norm,
        use_idf=use_idf,
        smooth_idf=smooth_idf,
        sublinear_tf=sublinear_tf,
    )

    res = tfidf.fit_transform(data_gpu).todense()
    ref = sk_tfidf.fit_transform(data).todense()

    cp.testing.assert_array_almost_equal(res, ref)


@pytest.mark.parametrize("norm", ["l1", "l2", None])
@pytest.mark.parametrize("use_idf", [True, False])
@pytest.mark.parametrize("smooth_idf", [True, False])
@pytest.mark.parametrize("sublinear_tf", [True, False])
def test_tfidf_transformer_copy(norm, use_idf, smooth_idf, sublinear_tf):
    if use_idf:
        pytest.xfail(
            "cupyx.scipy.sparse.csr does not support inplace multiply."
        )

    data_gpu = cupyx.scipy.sparse.csr_matrix(
        cp.array([[0, 1, 1, 1], [0, 2, 0, 1]], dtype=cp.float64, order="F")
    )

    tfidf = TfidfTransformer(
        norm=norm,
        use_idf=use_idf,
        smooth_idf=smooth_idf,
        sublinear_tf=sublinear_tf,
    )

    res = tfidf.fit_transform(data_gpu, copy=False)

    cp.testing.assert_array_almost_equal(data_gpu.todense(), res.todense())


def test_tfidf_transformer_sparse():
    X = cupyx.scipy.sparse.rand(10, 2000, dtype=np.float64, random_state=123)
    X_csc = cupyx.scipy.sparse.csc_matrix(X)
    X_csr = cupyx.scipy.sparse.csr_matrix(X)

    X_trans_csc = TfidfTransformer().fit_transform(X_csc).todense()
    X_trans_csr = TfidfTransformer().fit_transform(X_csr).todense()

    cp.testing.assert_array_almost_equal(X_trans_csc, X_trans_csr)
