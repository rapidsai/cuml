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
