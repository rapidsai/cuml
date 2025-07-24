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

import subprocess
import sys
from textwrap import dedent

import pytest

from cuml.accel.profilers import LineProfiler

pytest.importorskip("IPython")


SCRIPT_HEADER = """
from IPython.core.interactiveshell import InteractiveShell
from traitlets.config import Config
c = Config()
c.HistoryManager.hist_file = ":memory:"
ip = InteractiveShell(config=c)
"""


def run_script(body):
    script = SCRIPT_HEADER + dedent(body)

    res = subprocess.run(
        [sys.executable, "-c", script],
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        text=True,
    )
    # Pull out attributes before assert for nicer error reporting on failure
    returncode = res.returncode
    stdout = res.stdout
    assert returncode == 0, stdout
    return stdout


def test_magic():
    run_script(
        """
        ip.run_line_magic("load_ext", "cuml.accel")

        # cuml.accel proxies setup properly
        ip.run_cell("from sklearn.linear_model import LinearRegression")
        ip.run_cell("from cuml.accel import is_proxy")
        ip.run_cell("assert is_proxy(LinearRegression)").raise_error()
        """
    )


def test_magic_cudf_pandas_before():
    run_script(
        """
        ip.run_line_magic("load_ext", "cudf.pandas")
        ip.run_cell("import rmm; mr = rmm.mr.get_current_device_resource();")

        ip.run_line_magic("load_ext", "cuml.accel")
        ip.run_cell("mr2 = rmm.mr.get_current_device_resource();")

        # cuml doesn't change the mr setup by cudf.pandas
        ip.run_cell("assert mr is mr2").raise_error()

        # cuml.accel proxies setup properly
        ip.run_cell("from sklearn.linear_model import LinearRegression")
        ip.run_cell("from cuml.accel import is_proxy")
        result = ip.run_cell("assert is_proxy(LinearRegression)").raise_error()
        """
    )


def test_magic_cudf_pandas_after():
    run_script(
        """
        ip.run_line_magic("load_ext", "cuml.accel")
        ip.run_cell("import rmm; mr = rmm.mr.get_current_device_resource();")

        ip.run_line_magic("load_ext", "cudf.pandas")
        ip.run_cell("mr2 = rmm.mr.get_current_device_resource();")

        # cudf.pandas doesn't change the mr setup by cuml.accel
        ip.run_cell("assert mr is mr2").raise_error()

        # cuml.accel proxies setup properly
        ip.run_cell("from sklearn.linear_model import LinearRegression")
        ip.run_cell("from cuml.accel import is_proxy")
        result = ip.run_cell("assert is_proxy(LinearRegression)").raise_error()
        """
    )


@pytest.mark.parametrize(
    "magic", ["cuml.accel.profile", "cuml.accel.line_profile"]
)
def test_profiler_magics(magic):
    profiled_script = dedent(
        """
        from sklearn.datasets import make_regression
        from sklearn.linear_model import Ridge
        X, y = make_regression()
        model = Ridge().fit(X, y)
        """
    ).strip()

    stdout = run_script(
        f"""
        ip.run_line_magic("load_ext", "cuml.accel")

        script = {profiled_script!r}
        ip.run_cell_magic("{magic}", "", script)

        # Check that variables defined in the cell persist outside the cell
        ip.run_cell("assert isinstance(model, Ridge)").raise_error()

        # Check that the cell ran under cuml.accel
        ip.run_cell("from cuml.accel import is_proxy; assert is_proxy(model)").raise_error()

        # Check that the namespace modifications in LineProfiler don't persist
        ip.run_cell("assert {LineProfiler.FLAG!r} not in globals()").raise_error()
        """
    )
    if magic == "cuml.accel.profile":
        assert "cuml.accel profile" in stdout
    else:
        assert "cuml.accel line profile" in stdout
        assert profiled_script.splitlines()[0] in stdout
