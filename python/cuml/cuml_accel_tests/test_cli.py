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
import pickle
import pty
import subprocess
import sys
from textwrap import dedent

import pytest

from cuml.accel.__main__ import parse_args


def test_parse_no_args():
    ns = parse_args([])
    assert ns.script is None
    assert ns.module is None
    assert ns.verbose == 0
    assert ns.args == []


def test_parse_module():
    ns = parse_args(["-m", "mymodule"])
    assert ns.module == "mymodule"
    assert ns.script is None
    assert ns.args == []

    # trailing options are forwarded, even if they overlap with known options
    ns = parse_args(["-m", "mymodule", "-m", "more", "--unknown", "-v"])
    assert ns.module == "mymodule"
    assert ns.script is None
    assert ns.verbose == 0
    assert ns.args == ["-m", "more", "--unknown", "-v"]

    # earlier options do still apply
    ns = parse_args(["-v", "-m", "mymodule", "-v"])
    assert ns.module == "mymodule"
    assert ns.script is None
    assert ns.verbose == 1
    assert ns.args == ["-v"]


def test_parse_script():
    ns = parse_args(["script.py"])
    assert ns.module is None
    assert ns.script == "script.py"
    assert ns.args == []

    # trailing options are forwarded, even if they overlap with known options
    ns = parse_args(["script.py", "-m", "more", "--unknown", "-v"])
    assert ns.module is None
    assert ns.script == "script.py"
    assert ns.verbose == 0
    assert ns.args == ["-m", "more", "--unknown", "-v"]

    # earlier options do still apply
    ns = parse_args(["-v", "script.py", "-v"])
    assert ns.module is None
    assert ns.script == "script.py"
    assert ns.verbose == 1
    assert ns.args == ["-v"]


def test_parse_verbose():
    ns = parse_args(["-vvv"])
    assert ns.verbose == 3


def test_parse_format():
    ns = parse_args(["--format", "pickle"])
    assert ns.format == "pickle"

    ns = parse_args(["--format", "JOBLIB"])
    assert ns.format == "joblib"

    # Invalid formats error
    with pytest.raises(SystemExit) as exc:
        parse_args(["--format", "invalid"])
    assert exc.value.code != 0


def run(args=None, stdin=None, expected_returncode=0):
    proc = subprocess.Popen(
        [sys.executable, *map(str, args or [])],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    stdout, _ = proc.communicate(stdin)
    assert proc.returncode == expected_returncode, f"stdout:\n\n{stdout}"
    return stdout


SCRIPT = """
import cuml.accel
assert cuml.accel.enabled()

# Print here to assert the script actually ran by checking the output
print("ok")
"""


def test_cli_run_script(tmpdir):
    path = tmpdir.join("script.py")
    path.write(SCRIPT)
    stdout = run(["-m", "cuml.accel", path])
    assert "ok\n" in stdout


def test_cli_run_module(tmpdir):
    stdout = run(["-m", "cuml.accel", "-m", "code"], stdin=SCRIPT)
    assert "ok\n" in stdout


def test_cli_run_stdin():
    stdout = run(["-m", "cuml.accel"], stdin=SCRIPT)
    assert "ok\n" in stdout


def test_cli_run_stdin_errors():
    script = dedent(
        """
        print("got" + " here")
        assert False
        print("but" + " not here")
        """
    )
    stdout = run(["-m", "cuml.accel"], stdin=script, expected_returncode=1)
    assert "got here" in stdout
    assert "but not here" not in stdout
    assert "exec" not in stdout  # our exec not in traceback


def test_cli_run_interpreter():
    driver, receiver = pty.openpty()

    proc = subprocess.Popen(
        [sys.executable, "-m", "cuml.accel"],
        stdin=receiver,
        stdout=receiver,
        stderr=subprocess.STDOUT,
    )
    os.close(receiver)

    os.write(driver, b"import cuml.accel\n")
    os.write(driver, b"assert cuml.accel.enabled()\n")
    os.write(driver, b"print('got' + ' here')\n")
    os.write(driver, b"exit()\n")
    proc.wait(timeout=10)
    assert proc.returncode == 0

    stdout = os.read(driver, 10000).decode("utf-8")
    os.close(driver)
    assert "got here" in stdout
    assert "AssertionError" not in stdout


@pytest.mark.parametrize(
    "first, second",
    [("cuml.accel", "cudf.pandas"), ("cudf.pandas", "cuml.accel")],
)
def test_cli_mix_cuml_accel_and_cudf_pandas(first, second, tmpdir):
    script = dedent(
        """
        import sys
        assert sys.argv[1:] == ["-m", "custom"]

        import cuml.accel
        import cudf.pandas

        assert cuml.accel.enabled()
        assert cudf.pandas.LOADED

        # Print here to assert the script actually ran by checking the output
        print("ok")
        """
    )
    path = tmpdir.join("script.py")
    path.write(script)
    stdout = run(["-m", first, "-m", second, path, "-m", "custom"])
    assert "ok\n" in stdout


def test_cli_cudf_pandas():
    script = dedent(
        """
        import cuml.accel
        import cudf.pandas

        assert cuml.accel.enabled()
        assert cudf.pandas.LOADED

        # Print here to assert the script actually ran by checking the output
        print("ok")
        """
    )
    stdout = run(["-m", "cuml.accel", "--cudf-pandas"], stdin=script)
    assert "ok\n" in stdout


@pytest.mark.parametrize(
    "args, level", [([], "warn"), (["-v"], "info"), (["-vv"], "debug")]
)
def test_cli_verbose(args, level):
    script = dedent(
        f"""
        from cuml.internals.logger import get_level, level_enum
        level = get_level()
        expected = level_enum.{level}
        assert level == expected
        """
    )
    run(["-m", "cuml.accel", *args], stdin=script)


def test_cli_convert_to_sklearn(tmpdir):
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    X, y = make_classification(random_state=42)
    lr = LogisticRegression().fit(X, y)

    original = tmpdir.join("original.pkl")
    original.write(pickle.dumps(lr), mode="wb")
    output = tmpdir.join("output.pkl")

    script = dedent(
        f"""
        import cuml.accel
        import pickle

        with open({str(output)!r}, "rb") as f:
            new = pickle.load(f)

        assert not cuml.accel.is_proxy(new)
        """
    )

    # Run the conversion script
    run(
        [
            "-m",
            "cuml.accel",
            "--convert-to-sklearn",
            original,
            "--output",
            output,
        ]
    )

    # Check in a new process if the reloaded estimator is a proxy estimator
    run(stdin=script)
