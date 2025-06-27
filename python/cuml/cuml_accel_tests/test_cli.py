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
import re
import subprocess
import sys
from textwrap import dedent

import pytest

from cuml.accel.__main__ import parse_args


def test_parse_no_args():
    ns = parse_args([])
    assert ns.script == "-"
    assert ns.cmd is None
    assert ns.module is None
    assert ns.verbose == 0
    assert ns.args == []


def test_parse_explicit_stdin():
    ns = parse_args(["-v", "-", "-vv"])
    assert ns.script == "-"
    assert ns.cmd is None
    assert ns.module is None
    assert ns.verbose == 1
    assert ns.args == ["-vv"]


def test_parse_module():
    ns = parse_args(["-m", "mymodule"])
    assert ns.module == "mymodule"
    assert ns.script == "-"
    assert ns.cmd is None
    assert ns.args == []

    # trailing options are forwarded, even if they overlap with known options
    ns = parse_args(["-m", "mymodule", "-m", "more", "--unknown", "-v"])
    assert ns.module == "mymodule"
    assert ns.script == "-"
    assert ns.cmd is None
    assert ns.verbose == 0
    assert ns.args == ["-m", "more", "--unknown", "-v"]

    # earlier options do still apply
    ns = parse_args(["-v", "-m", "mymodule", "-v"])
    assert ns.module == "mymodule"
    assert ns.script == "-"
    assert ns.cmd is None
    assert ns.verbose == 1
    assert ns.args == ["-v"]


def test_parse_cmd():
    ns = parse_args(["-c", "print('hello')"])
    assert ns.module is None
    assert ns.script == "-"
    assert ns.cmd == "print('hello')"
    assert ns.args == []

    # trailing options are forwarded, even if they overlap with known options
    ns = parse_args(["-c", "print('hello')", "-c", "more", "--unknown", "-v"])
    assert ns.module is None
    assert ns.script == "-"
    assert ns.cmd == "print('hello')"
    assert ns.verbose == 0
    assert ns.args == ["-c", "more", "--unknown", "-v"]

    # earlier options do still apply
    ns = parse_args(["-v", "-c", "print('hello')", "-v"])
    assert ns.module is None
    assert ns.script == "-"
    assert ns.cmd == "print('hello')"
    assert ns.verbose == 1
    assert ns.args == ["-v"]


def test_parse_choses_first_c_or_m():
    ns = parse_args(["-c", "print('hello')", "-m", "foo"])
    assert ns.cmd == "print('hello')"
    assert ns.module is None
    assert ns.args == ["-m", "foo"]

    ns = parse_args(["-m", "foo", "-c", "print('hello')"])
    assert ns.cmd is None
    assert ns.module == "foo"
    assert ns.args == ["-c", "print('hello')"]


def test_parse_script():
    ns = parse_args(["script.py"])
    assert ns.module is None
    assert ns.script == "script.py"
    assert ns.cmd is None
    assert ns.args == []

    # trailing options are forwarded, even if they overlap with known options
    ns = parse_args(["script.py", "-m", "more", "--unknown", "-v"])
    assert ns.module is None
    assert ns.script == "script.py"
    assert ns.cmd is None
    assert ns.verbose == 0
    assert ns.args == ["-m", "more", "--unknown", "-v"]

    # earlier options do still apply
    ns = parse_args(["-v", "script.py", "-v"])
    assert ns.module is None
    assert ns.script == "script.py"
    assert ns.cmd is None
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


def test_cli_run_cmd():
    stdout = run(["-m", "cuml.accel", "-c", SCRIPT])
    assert "ok\n" in stdout


@pytest.mark.parametrize("pass_hyphen", [False, True])
def test_cli_run_stdin(pass_hyphen):
    args = ["-m", "cuml.accel"]
    if pass_hyphen:
        args.append("-")
    stdout = run(args, stdin=SCRIPT)
    assert "ok\n" in stdout


@pytest.mark.parametrize("mode", ["script", "module", "cmd", "stdin"])
def test_cli_correct_argv(mode, tmpdir):
    """Test that user code sees the same argv with and and without `cuml.accel`"""
    script = "import sys;print(f'argv={sys.argv}')"
    stdin = None
    if mode == "script":
        path = tmpdir.join("script.py")
        path.write(script)
        args = [path, "--foo"]
    elif mode == "module":
        args = ["-m", "code", "-q"]
        stdin = script
    elif mode == "cmd":
        args = ["-c", script, "--foo"]
    else:
        args = ["-", "--foo"]
        stdin = script

    stdout = run(args, stdin=stdin)
    stdout_accel = run(["-m", "cuml.accel", *args], stdin=stdin)

    argv = re.search(r"argv=\[.*\]", stdout).group()
    argv_accel = re.search(r"argv=\[.*\]", stdout_accel).group()
    assert argv == argv_accel


@pytest.mark.parametrize("mode", ["stdin", "cmd", "script"])
def test_cli_run_errors(mode, tmpdir):
    script = dedent(
        """
        print("got" + " here")
        assert False
        print("but" + " not here")
        """
    )
    if mode == "stdin":
        args = ["-m", "cuml.accel"]
        stdin = script
    elif mode == "cmd":
        args = ["-m", "cuml.accel", "-c", script]
        stdin = None
    else:
        path = tmpdir.join("script.py")
        path.write(script)
        args = ["-m", "cuml.accel", path]
        stdin = None

    stdout = run(args, stdin=stdin, expected_returncode=1)
    assert "got here" in stdout
    assert "but not here" not in stdout
    if mode in ("stdin", "cmd"):
        assert "exec" not in stdout  # our exec not in traceback


def test_cli_run_interpreter():
    driver_fd, receiver_fd = pty.openpty()

    proc = subprocess.Popen(
        [sys.executable, "-m", "cuml.accel"],
        stdin=receiver_fd,
        stdout=receiver_fd,
        stderr=subprocess.STDOUT,
    )
    os.close(receiver_fd)

    driver = os.fdopen(driver_fd, mode="a")
    driver.write("import cuml.accel\n")
    driver.write("assert cuml.accel.enabled()\n")
    driver.write("print('got' + ' here')\n")
    driver.write("exit()\n")

    proc.wait(timeout=20)
    assert proc.returncode == 0

    stdout = os.read(driver_fd, 10000).decode("utf-8")
    driver.close()
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
