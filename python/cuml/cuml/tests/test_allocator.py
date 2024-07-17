#
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

from cuml.internals.input_utils import sparse_scipy_to_cp
from cuml.testing.utils import small_classification_dataset
from cuml.naive_bayes import MultinomialNB
from cuml import LogisticRegression
from cuml.internals.safe_imports import cpu_only_import
import pytest

from cuml.internals.safe_imports import gpu_only_import

cp = gpu_only_import("cupy")
np = cpu_only_import("numpy")


try:
    from cupy.cuda import using_allocator as cupy_using_allocator
except ImportError:
    from cupy.cuda.memory import using_allocator as cupy_using_allocator


def dummy_allocator(nbytes):
    raise AssertionError("Dummy allocator should not be called")


def test_dummy_allocator():
    with pytest.raises(AssertionError):
        with cupy_using_allocator(dummy_allocator):
            a = cp.arange(10)
            del a


def test_logistic_regression():
    with cupy_using_allocator(dummy_allocator):
        X_train, X_test, y_train, y_test = small_classification_dataset(
            np.float32
        )
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)
        culog = LogisticRegression()
        culog.fit(X_train, y_train)
        culog.predict(X_train)


def test_naive_bayes(nlp_20news):
    X, y = nlp_20news

    X = sparse_scipy_to_cp(X, cp.float32).astype(cp.float32)
    y = y.astype(cp.int32)

    with cupy_using_allocator(dummy_allocator):
        model = MultinomialNB()
        model.fit(X, y)

        y_hat = model.predict(X)
        y_hat = model.predict(X)
        y_hat = model.predict_proba(X)
        y_hat = model.predict_log_proba(X)
        y_hat = model.score(X, y)

        del y_hat
