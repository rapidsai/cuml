# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import print_function
import os
import subprocess
import argparse
import time
import re
import sys


ToolOptions = {
    "memcheck": "--leak-check=full",
    "initcheck": "--track-unused-memory=yes",
    "racecheck": "",
    "synccheck": "",
}
CommentRegex = re.compile(r"\s*#.+")


def parse_args():
    argparser = argparse.ArgumentParser(
        "Runs googletest unit-tests with compute-sanitizer enabled"
    )
    argparser.add_argument(
        "-exe",
        type=str,
        default=None,
        help="The googletest executable to be run",
    )
    argparser.add_argument(
        "-pwd",
        type=str,
        default=None,
        help="Current directory for running the exe",
    )
    argparser.add_argument(
        "-tool",
        type=str,
        default="memcheck",
        choices=["memcheck", "initcheck", "racecheck", "synccheck"],
        help="memcheck tool to be used",
    )
    argparser.add_argument(
        "-v",
        dest="verbose",
        action="store_true",
        help="Print verbose messages",
    )
    args = argparser.parse_args()
    if args.exe is None:
        raise Exception("'-exe' is a mandatory option!")
    return args


def run_cmd(cmd, workdir):
    cwd = os.getcwd()
    if workdir:
        os.chdir(workdir)
    result = subprocess.run(
        cmd,
        check=False,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    out = result.stdout.decode("utf-8").rstrip()
    if workdir:
        os.chdir(cwd)
    return result.returncode, out


def get_testlist(exe, workdir):
    retcode, out = run_cmd("%s --gtest_list_tests" % exe, workdir)
    if retcode != 0:
        print(out)
        raise Exception("Collecting test-list failed! See above errors")
    tests = []
    for line in out.splitlines():
        # only consider fixtures and main tests
        if line[:2] == "  " or line.startswith("Running "):
            continue
        line = CommentRegex.sub("", line)
        tests.append(line + "*")
    return tests


def run_tests(args, testlist):
    start = time.time()
    idx = 1
    failed = []
    total = len(testlist)
    for test in testlist:
        cmd = "compute-sanitizer --require-cuda-init=no --tool %s %s %s " % (
            args.tool,
            ToolOptions[args.tool],
            args.exe,
        )
        cmd += "--gtest_filter=%s" % test
        print(
            "[%d/%d Failed:%d] Checking %s ... "
            % (idx, total, len(failed), test),
            end="",
        )
        sys.stdout.flush()
        retcode, out = run_cmd(cmd, args.pwd)
        print("[%s]" % ("PASS" if retcode == 0 else "FAIL"))
        if retcode != 0:
            if args.verbose:
                print(out)
            failed.append(test)
        idx += 1
    if len(failed) != 0:
        print(
            "FAIL: %d failed tests out of %d. Failed tests are"
            % (len(failed), total)
        )
        for f in failed:
            print("  %s" % f)
    else:
        print("PASS")
    diff = time.time() - start
    print("Total time taken: %d s" % diff)
    if len(failed) != 0:
        raise Exception("Test failed!")


def main():
    args = parse_args()
    testlist = get_testlist(args.exe, args.pwd)
    run_tests(args, testlist)
    return


if __name__ == "__main__":
    main()
