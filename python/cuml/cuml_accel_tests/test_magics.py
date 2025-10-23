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
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from IPython.core.interactiveshell import InteractiveShell
from traitlets.config import Config
c = Config()
c.HistoryManager.hist_file = ":memory:"
c.InteractiveShell.colors = "nocolor"
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
    "magics",
    [
        ("cuml.accel.profile",),
        ("cuml.accel.line_profile",),
        ("cuml.accel.profile", "cuml.accel.line_profile"),
        ("cuml.accel.line_profile", "cuml.accel.profile"),
        ("cuml.accel.line_profile", "cuml.accel.profile", "time"),
    ],
)
def test_profiler_magics(magics):
    """Check the profiler magics work and can be combined in any order"""
    magic_lines = [f"%%{magic}" for magic in magics]
    cell = "\n".join(
        [
            *magic_lines,
            "from sklearn.datasets import make_regression",
            "from sklearn.linear_model import Ridge",
            "X, y = make_regression()",
            "model = Ridge().fit(X, y)",
            "'something ' + 'to ' + 'output'",
        ]
    )

    stdout = run_script(
        f"""
        ip.run_line_magic("load_ext", "cuml.accel")

        ip.run_cell({cell!r})

        # Check that the output of the cell exists
        ip.run_cell("assert _ == 'something to output'").raise_error()

        # Check that variables defined in the cell persist outside the cell
        ip.run_cell("assert isinstance(model, Ridge)").raise_error()

        # Check that the cell ran under cuml.accel
        ip.run_cell("from cuml.accel import is_proxy; assert is_proxy(model)").raise_error()
        """
    )
    # Get the index of the output cell to assert that the profiler output
    # renders _before_ the output cell.
    output_index = stdout.index("Out[1]:")

    # Check that the output of the cell is displayed
    assert "something to output" in stdout

    if "cuml.accel.profile" in magics:
        assert "cuml.accel profile" in stdout
        assert stdout.rindex("cuml.accel profile") < output_index
    if "cuml.accel.line_profile" in magics:
        assert "cuml.accel line profile" in stdout
        assert stdout.rindex("cuml.accel line profile") < output_index
        assert "%%cuml.accel" not in stdout
        assert "from sklearn.datasets import make_regression" in stdout
        # Ensure the line profile includes the original script and not
        # the transformed one with `get_ipython`
        assert "get_ipython()" not in stdout


def test_profiler_magics_output():
    """Check the profiler magic handles outputs from cells properly"""
    # Multi-line interactive cell
    cell1 = dedent(
        """
        %%cuml.accel.profile
        a = 1
        f"string from cell {a}"
        """
    ).strip()

    # Single-line interactive cell
    cell2 = dedent(
        """
        %%cuml.accel.profile
        f"string from cell {a + 1}"
        """
    )

    # Not interactive, silenced by semicolon
    cell3 = dedent(
        """
        %%cuml.accel.profile
        f"string from cell {a + 2}";
        """
    ).strip()

    # Not interactive, statement comes after expression
    cell4 = dedent(
        """
        %%cuml.accel.profile
        f"string from cell {a + 3}"
        a += 3
        """
    )

    # Not interactive, only statements
    cell5 = dedent(
        """
        %%cuml.accel.profile
        b = a + 1
        """
    )

    stdout = run_script(
        f"""
        ip.run_line_magic("load_ext", "cuml.accel")

        ip.run_cell({cell1!r}, store_history=True)
        ip.run_cell("assert _ == 'string from cell 1'").raise_error()

        ip.run_cell({cell2!r}, store_history=True)
        ip.run_cell("assert _ == 'string from cell 2'").raise_error()

        ip.run_cell({cell3!r}, store_history=True)
        ip.run_cell("assert _ == 'string from cell 2'").raise_error()

        ip.run_cell({cell4!r}, store_history=True)
        ip.run_cell("assert _ == 'string from cell 2'").raise_error()

        ip.run_cell({cell5!r}, store_history=True)
        ip.run_cell("assert _ == 'string from cell 2'").raise_error()
        """
    )
    assert "Out[1]:" in stdout
    assert "Out[2]:" in stdout
    for cell in [3, 4, 5]:
        assert f"Out[{cell}]:" not in stdout
        assert f"string from cell {cell}" not in stdout


def test_profiler_magics_error_before_exec():
    stdout = run_script(
        """
        ip.run_line_magic("load_ext", "cuml.accel")

        script = '''
        %%cuml.accel.profile
        print('got ' + 'here')
        this is not valid python
        '''

        status = ip.run_cell(script)
        assert isinstance(status.error_in_exec, SyntaxError)
        """
    )
    assert "got here" not in stdout
    # Table not rendered
    assert "cuml.accel profile" not in stdout
    # Only one traceback rendered
    assert stdout.count("Traceback (most recent call last)") == 1


def test_profiler_magics_error_during_exec():
    stdout = run_script(
        """
        ip.run_line_magic("load_ext", "cuml.accel")

        script = '''
        %%cuml.accel.profile
        print('got ' + 'here')
        raise ValueError("oh no!")
        '''

        status = ip.run_cell(script)
        assert isinstance(status.error_in_exec, ValueError)
        """
    )
    assert "got here" in stdout
    # Table still rendered
    assert "cuml.accel profile" in stdout
    # Only one traceback rendered
    assert stdout.count("Traceback (most recent call last)") == 1


def test_line_profiler_magic_applies_to_the_correct_source():
    """Test that only selected lines are profiled, and that the source lines
    are properly matched to the execution lines."""
    stdout = run_script(
        """
        ip.run_line_magic("load_ext", "cuml.accel")

        cell1 = '''
        def run_fit(model, X, y):
            # A function defined in an earlier cell, shouldn't be traced
            model.fit(X, y)
            return model
        '''

        cell2 = '''


        %%cuml.accel.line_profile
        %%cuml.accel.profile
        from sklearn.datasets import make_regression
        from sklearn.linear_model import Ridge
        X, y = make_regression()
        run_fit(Ridge(), X, y)
        model = Ridge().fit(X, y)


        '''

        ip.run_cell(cell1).raise_error()
        ip.run_cell(cell2).raise_error()
        """
    )
    # Rows in the line profiler table.
    rows = (
        stdout.split("cuml.accel line profile")[-1].strip().splitlines()[3:-2]
    )
    assert len(rows) == 5
    counts = []
    gpu_used = []
    for row in rows:
        parts = row.split("│")[1:-1]
        counts.append(int(parts[1].strip()))
        gpu_used.append(parts[3].strip() != "-")
    assert counts == [1, 1, 1, 1, 1]
    assert gpu_used == [False, False, False, True, True]
