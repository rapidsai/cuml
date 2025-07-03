#
# Copyright (c) 2025, NVIDIA CORPORATION.
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
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge

from cuml.accel.core import logger


@pytest.fixture(autouse=True)
def reset_level():
    """A fixture to reset the log level back to default after every test"""
    yield
    logger.set_level("warn")


@pytest.fixture
def get_logs(capsys):
    """A fixture to get the logs output by a test"""

    def get_logs():
        out, _ = capsys.readouterr()
        return [
            line for line in out.split("\n") if line.startswith("[cuml.accel]")
        ]

    return get_logs


@pytest.mark.parametrize(
    "log_level,expected",
    [
        (
            "warn",
            [],
        ),
        (
            "info",
            [
                "[cuml.accel] `Ridge.fit` ran on GPU",
                "[cuml.accel] `Ridge.fit` ran on GPU",
                "[cuml.accel] `Ridge.predict` ran on GPU",
            ],
        ),
        (
            "debug",
            [
                "[cuml.accel] `Ridge.fit` ran on GPU",
                "[cuml.accel] `Ridge` fitted attributes synced to CPU",
                "[cuml.accel] `Ridge` parameters synced to GPU",
                "[cuml.accel] `Ridge.fit` ran on GPU",
                "[cuml.accel] `Ridge.predict` ran on GPU",
            ],
        ),
    ],
)
def test_logging_multiple_operations(get_logs, log_level, expected):
    logger.set_level(log_level)

    X, y = make_regression(random_state=42)
    ridge = Ridge(random_state=42)
    ridge.fit(X, y)
    _ = ridge.coef_
    ridge.alpha = 2.0
    ridge.fit(X, y)
    ridge.predict(X)

    assert get_logs() == expected


def test_unsupported_hyperparams(get_logs):
    logger.set_level("info")

    X, y = make_regression(random_state=42)
    ridge = Ridge(random_state=42, positive=True)
    ridge.fit(X, y)
    ridge.predict(X)

    expected = [
        "[cuml.accel] `Ridge.fit` falling back to CPU: `positive=True` is not supported",
        "[cuml.accel] `Ridge.fit` ran on CPU",
        "[cuml.accel] `Ridge.predict` ran on CPU",
    ]
    assert get_logs() == expected


@pytest.mark.parametrize("log_level", ["info", "debug"])
def test_unsupported_hyperparams_in_set_params(get_logs, log_level):
    logger.set_level(log_level)

    X, y = make_regression(random_state=42)
    ridge = Ridge(random_state=42)
    ridge.fit(X, y)
    # Triggers a fallback to CPU
    ridge.set_params(positive=True)
    ridge.predict(X)

    expected = [
        "[cuml.accel] `Ridge.fit` ran on GPU",
        (
            "[cuml.accel] `Ridge` parameters failed to sync to GPU, falling back to "
            "CPU: `positive=True` is not supported"
        ),
        "[cuml.accel] `Ridge` fitted attributes synced to CPU",
        "[cuml.accel] `Ridge.predict` ran on CPU",
    ]
    if log_level == "info":
        del expected[2]  # only logged at debug
    assert get_logs() == expected


def test_unsupported_parameters(get_logs):
    logger.set_level("info")

    X, y = make_regression(random_state=42, n_targets=2)
    ridge = Ridge(random_state=42)
    ridge.fit(X, y)
    ridge.predict(X)

    expected = [
        "[cuml.accel] `Ridge.fit` falling back to CPU: Multioutput `y` is not supported",
        "[cuml.accel] `Ridge.fit` ran on CPU",
        "[cuml.accel] `Ridge.predict` ran on CPU",
    ]
    assert get_logs() == expected
