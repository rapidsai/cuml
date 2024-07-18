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
from cuml import LogisticRegression as cuLog
from cuml import multiclass as cu_multiclass
import sys
import pytest
from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import("numpy")


# As tests directory is not a module, we need to add it to the path
sys.path.insert(0, ".")
from test_linear_model import make_classification_dataset  # noqa: E402


@pytest.mark.parametrize("strategy", ["ovr", "ovo"])
@pytest.mark.parametrize("use_wrapper", [True, False])
@pytest.mark.parametrize("nrows", [1000])
@pytest.mark.parametrize("num_classes", [3])
@pytest.mark.parametrize("column_info", [[10, 4]])
def test_logistic_regression(
    strategy, use_wrapper, nrows, num_classes, column_info, dtype=np.float32
):

    ncols, n_info = column_info

    X_train, X_test, y_train, y_test = make_classification_dataset(
        datatype=dtype,
        nrows=nrows,
        ncols=ncols,
        n_info=n_info,
        num_classes=num_classes,
    )
    y_train = y_train.astype(dtype)
    y_test = y_test.astype(dtype)
    culog = cuLog()

    if use_wrapper:
        cls = cu_multiclass.MulticlassClassifier(culog, strategy=strategy)
    else:
        if strategy == "ovo":
            cls = cu_multiclass.OneVsOneClassifier(culog)
        else:
            cls = cu_multiclass.OneVsRestClassifier(culog)

    cls.fit(X_train, y_train)
    test_score = cls.score(X_test, y_test)
    assert test_score > 0.7
