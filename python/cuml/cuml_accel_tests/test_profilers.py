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

import os
import textwrap

import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import Ridge
from sklearn.svm import SVC

from cuml.accel.profilers import LineProfiler, format_duration, profile


@pytest.fixture
def wide_terminal(monkeypatch):
    """Ensure rich's output won't truncate columns in our tests"""
    monkeypatch.setenv("COLUMNS", "100")


def line_profile(source: str) -> LineProfiler:
    filename = f"<cuml-accel-input-{os.urandom(6).hex()}>"
    code = compile(source, filename, "exec")
    with LineProfiler(source=source, filename=filename, quiet=True) as lprof:
        exec(code, {"__name__": "__main__"})
    return lprof


def test_format_duration():
    assert format_duration(0) == "0s"
    assert format_duration(0.0001201) == "120.1µs"
    assert format_duration(0.0001) == "100µs"
    assert format_duration(0.0012) == "1.2ms"
    assert format_duration(0.01) == "10ms"
    assert format_duration(0.1) == "100ms"
    assert format_duration(0.1234) == "123.4ms"
    assert format_duration(12.3456) == "12.3s"
    assert format_duration(12.0) == "12s"
    assert format_duration(65) == "1m5s"
    assert format_duration(65.2345) == "1m5.2s"
    assert format_duration(3600) == "1h0m0s"
    assert format_duration(3601.234) == "1h0m1.2s"
    assert format_duration(3661.234) == "1h1m1.2s"


def test_line_profile(capsys, wide_terminal):
    script = textwrap.dedent(
        """
        from sklearn.linear_model import Ridge
        from sklearn.datasets import make_regression

        X, y = make_regression(n_samples=10000)

        # Fit and predict. The first 2 will run on GPU, the third on CPU
        for params in [{"alpha": 1.0}, {"alpha": 2.0}, {"positive": True}]:
            ridge = Ridge(**params)
            ridge.fit(X, y)
            ridge.predict(X)
        """
    ).strip()
    lprof = line_profile(script)
    assert lprof.total_gpu_time > 0
    assert lprof.total_time > lprof.total_gpu_time

    # A line that uses GPU (`ridge.fit` call)
    line_9 = lprof.line_stats[9]
    assert line_9.gpu_time > 0
    assert line_9.count == 3
    assert line_9.total_time > line_9.gpu_time
    assert line_9.cpu_fallback

    # A line that cannot use GPU
    assert 1 in lprof.line_stats
    line_1 = lprof.line_stats[1]
    assert line_1.count == 1
    assert line_1.gpu_time == 0

    # A line that contains no runnable source isn't tracked
    assert 3 not in lprof.line_stats

    # Smoketest the output report
    lprof.print_report()
    out, _ = capsys.readouterr()
    for col in ["#", "N", "Time", "GPU %", "Source"]:
        assert col in out
    assert script.splitlines()[0] in out


def test_line_profile_nested(capsys, wide_terminal):
    script = textwrap.dedent(
        """
        from time import sleep
        def add(x, y):
            sleep(0.1)
            return x + y

        def inc(x):
            return add(x, 1)

        a = 1
        b = 2
        c = add(a, b)
        d = inc(c)
        e = add(c, d)
        """
    ).strip()

    lprof = line_profile(script)
    assert lprof.total_gpu_time == 0
    assert lprof.total_time > 0.3

    sleep_line = lprof.line_stats[3]
    add_1_line = lprof.line_stats[7]
    add_a_b_line = lprof.line_stats[11]

    assert sleep_line.count == 3
    assert sleep_line.total_time >= 0.3
    assert add_1_line.count == 1
    assert add_1_line.total_time >= 0.1
    assert add_a_b_line.count == 1
    assert add_a_b_line.total_time >= 0.1

    # Smoketest the output report works with no GPU usage
    lprof.print_report()
    out, _ = capsys.readouterr()
    assert "0.0% on GPU" in out


def test_line_profile_errors():
    script = textwrap.dedent(
        """
        from sklearn.linear_model import Ridge
        from sklearn.datasets import make_regression

        X, y = make_regression(n_samples=100000)
        model = Ridge().fit(X, y)
        raise ValueError("Oh no!")
        model.predict(X)
        """
    ).strip()

    filename = f"<cuml-accel-input-{os.urandom(6).hex()}>"
    code = compile(script, filename, "exec")
    with pytest.raises(ValueError, match="Oh no!"):
        with LineProfiler(
            source=script, filename=filename, quiet=True
        ) as lprof:
            exec(code, {"__name__": "__main__"})

    # Timers properly unwound
    assert not lprof._timers
    assert lprof.line_stats[5].gpu_time > 0.0
    # Erroring line still counted and timed
    assert lprof.line_stats[6].count == 1
    assert lprof.line_stats[6].total_time > 0.0
    # Line that was never ran not present
    assert 7 not in lprof.line_stats


def test_profile(capsys, wide_terminal):
    X, y = make_regression(n_samples=10000)

    with profile(quiet=True) as results:
        # Fit and predict. The first 2 will run on GPU, the third on CPU
        for params in [{"alpha": 1.0}, {"alpha": 2.0}, {"positive": True}]:
            ridge = Ridge(**params)
            ridge.fit(X, y)
            ridge.predict(X)

    assert set(results.method_calls) == {"Ridge.fit", "Ridge.predict"}
    # Check stats for a method make sense
    fit_stats = results.method_calls["Ridge.fit"]
    assert fit_stats.gpu_calls == 2
    assert fit_stats.gpu_time > 0
    assert fit_stats.cpu_calls == 1
    assert fit_stats.cpu_time > 0
    assert len(fit_stats.fallback_reasons) == 1

    # Smoketest the output report
    results.print_report()
    out, _ = capsys.readouterr()
    for method in results.method_calls:
        assert method in out
    assert "Not all operations ran on the GPU" in out
    assert list(fit_stats.fallback_reasons)[0] in out


def test_profile_fallback_in_gpu_method():
    X, y = make_classification(n_classes=4, n_informative=4)
    model = SVC()
    with profile(quiet=True) as results:
        # Hyperparameters supported but method args aren't
        model.fit(X, y)

    fit_stats = results.method_calls["SVC.fit"]
    assert fit_stats.gpu_calls == 0
    assert fit_stats.gpu_time == 0
    assert fit_stats.cpu_calls == 1
    assert fit_stats.cpu_time > 0
    assert len(fit_stats.fallback_reasons) == 1
