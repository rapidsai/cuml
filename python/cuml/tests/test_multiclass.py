# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import pytest

from cuml import LogisticRegression as cuLog
from cuml import multiclass as cu_multiclass
from cuml.testing.datasets import make_classification_dataset


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
