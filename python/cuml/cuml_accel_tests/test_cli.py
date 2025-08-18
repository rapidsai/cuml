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


def run(args=None, stdin=None, env=None, expected_returncode=0):
    # Run without `CUML_ACCEL_ENABLED` defined by default to test
    # the other accelerator loading mechanisms
    proc_env = os.environ.copy()
    proc_env.pop("CUML_ACCEL_ENABLED", None)
    if env is not None:
        proc_env.update(env)

    proc = subprocess.Popen(
        [sys.executable, *map(str, args or [])],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        env=proc_env,
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


@pytest.mark.parametrize("value", ["1", "TrUe", "0", "", None])
def test_cuml_accel_enabled_environ(value):
    script = dedent(
        """
        import cuml
        import os
        print(os.environ)
        print("ENABLED" if cuml.accel.enabled() else "DISABLED")
        """
    )
    env = {}
    if value is not None:
        env["CUML_ACCEL_ENABLED"] = value
    expected = (
        "ENABLED" if (value or "").lower() in ("1", "true") else "DISABLED"
    )
    stdout = run([], stdin=script, env=env)
    assert expected in stdout


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

    env = os.environ.copy()
    env.pop("CUML_ACCEL_ENABLED", None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "cuml.accel"],
        stdin=receiver_fd,
        stdout=receiver_fd,
        stderr=subprocess.STDOUT,
        env=env,
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


@pytest.mark.parametrize(
    "args, level", [([], "warn"), (["-v"], "info"), (["-vv"], "debug")]
)
def test_cli_verbose(args, level):
    script = dedent(
        f"""
        from cuml.accel.core import logger
        level = logger.level.name.lower()
        assert level == {level!r}
        """
    )
    run(["-m", "cuml.accel", *args], stdin=script)


@pytest.mark.parametrize("mode", ["script", "module", "cmd", "stdin"])
@pytest.mark.parametrize(
    "options",
    [
        ("--profile",),
        ("--line-profile",),
        (
            "--profile",
            "--line-profile",
        ),
    ],
)
def test_cli_profilers(mode, options, tmpdir):
    """Check that the --profile and --line-profile options work across execution modes."""
    script = dedent(
        """
        from sklearn.datasets import make_regression
        from sklearn.linear_model import Ridge

        X, y = make_regression()
        model = Ridge().fit(X, y)
        model.predict(X)
        """
    ).strip()

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

    if "--line-profile" in options and mode == "module":
        # Check the case where --line-profile isn't supported
        stdout = run(
            ["-m", "cuml.accel", *options, *args],
            stdin=stdin,
            expected_returncode=1,
        )
        assert "--line-profile is not supported with -m" in stdout
        return

    stdout = run(["-m", "cuml.accel", *options, *args], stdin=stdin)

    if "--line-profile" in options:
        assert "cuml.accel line profile" in stdout
        assert script.splitlines()[0] in stdout

    if "--profile" in options:
        if "--line-profile" in options:
            # Check that there's a blank line between the reports
            assert "\n\ncuml.accel profile" in stdout
        else:
            assert "cuml.accel profile" in stdout
        assert "Ridge.fit" in stdout
        assert "Ridge.predict" in stdout


@pytest.mark.parametrize("mode", ["script", "cmd", "stdin"])
@pytest.mark.parametrize("option", ["--profile", "--line-profile"])
def test_cli_profilers_errors(mode, option, tmpdir):
    """Test that the profile output is still rendered if the script errors"""
    script = dedent(
        """
        print("got" + " here")
        assert False
        print("but" + " not here")
        """
    ).strip()
    if mode == "stdin":
        args = ["-m", "cuml.accel", option]
        stdin = script
    elif mode == "cmd":
        args = ["-m", "cuml.accel", option, "-c", script]
        stdin = None
    else:
        path = tmpdir.join("script.py")
        path.write(script)
        args = ["-m", "cuml.accel", option, path]
        stdin = None

    stdout = run(args, stdin=stdin, expected_returncode=1)
    assert "got here" in stdout
    if option == "--profile":
        assert "cuml.accel profile" in stdout
    if option == "--line-profile":
        assert "cuml.accel line profile" in stdout
    assert "but not here" not in stdout
    if mode in ("stdin", "cmd"):
        assert "exec" not in stdout  # our exec not in traceback
