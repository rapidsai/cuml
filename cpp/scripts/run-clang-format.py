# Copyright (c) 2019, NVIDIA CORPORATION.
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

from __future__ import print_function
import sys
import re
import os
import subprocess
import argparse


VersionRegex = re.compile(r"clang-format version ([0-9.]+)")


def parse_args():
    argparser = argparse.ArgumentParser("Runs clang-format on a project")
    argparser.add_argument("-dstdir", type=str,
                           default="/tmp/cuml-clang-format",
                           help="Path to the current build directory.")
    argparser.add_argument("-exe", type=str, default="clang-format",
                           help="Path to clang-format exe")
    argparser.add_argument("-inplace", default=False, action="store_true",
                           help="Replace the source files itself.")
    argparser.add_argument("-regex", type=str,
                           default=r"[.](cu|cuh|h|hpp|cpp)$",
                           help="Regex string to filter in sources")
    argparser.add_argument("-ignore", type=str, default=r"cannylab/bh[.]cu$",
                           help="Regex used to ignore files from matched list")
    argparser.add_argument("dirs", type=str, nargs="*",
                           help="List of dirs where to find sources")
    args = argparser.parse_args()
    args.regexCompiled = re.compile(args.regex)
    args.ignoreCompiled = re.compile(args.ignore)
    ret = subprocess.check_output("%s --version" % args.exe, shell=True)
    ret = ret.decode("utf-8")
    version = VersionRegex.match(ret)
    if version is None:
        raise Exception("Failed to figure out clang-format version!")
    version = version.group(1)
    if version != "8.0.0":
        raise Exception("clang-format exe must be v8.0.0 found '%s'" % version)
    return args


def list_all_src_files(fileRegex, ignoreRegex, srcdirs, dstdir, inplace):
    allFiles = []
    for srcdir in srcdirs:
        for root, dirs, files in os.walk(srcdir):
            for f in files:
                if re.search(fileRegex, f):
                    src = os.path.join(root, f)
                    if re.search(ignoreRegex, src):
                        continue
                    if inplace:
                        _dir = root
                    else:
                        _dir = os.path.join(dstdir, os.path.basename(root))
                    dst = os.path.join(_dir, f)
                    allFiles.append((src, dst))
    return allFiles


def run_clang_format(src, dst, exe):
    dstdir = os.path.dirname(dst)
    if not os.path.exists(dstdir):
        os.makedirs(dstdir)
    # run the clang format command itself
    if src == dst:
        cmd = "%s -i %s" % (exe, src)
    else:
        cmd = "%s %s > %s" % (exe, src, dst)
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError:
        print("Failed to run clang-format! Maybe your env is not proper?")
        raise
    # run the diff to check if there are any formatting issues
    cmd = "diff -q %s %s >/dev/null" % (src, dst)
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError:
        srcPath = os.path.join(os.getcwd(), src)
        print("clang-format failed! Run 'diff -y %s %s' to know more about "
              "the formatting violations!" % (srcPath, dst))
        return False
    return True


def main():
    args = parse_args()
    allFiles = list_all_src_files(args.regexCompiled, args.ignoreCompiled,
                                  args.dirs, args.dstdir, args.inplace)
    # actual format checker
    status = True
    for src, dst in allFiles:
        if not run_clang_format(src, dst, args.exe):
            status = False
    if not status:
        print("Clang-format failed! You have 2 options:")
        print(" 1. Look at formatting differences above and fix them manually")
        print(" 2. Or run the below command to bulk-fix all these at once")
        print("Bulk-fix command: ")
        print("  cd /path/to/your/cuml/repo")
        print("  python cpp/scripts/run-clang-format.py %s -inplace" % \
              " ".join(sys.argv[1:]))
        sys.exit(-1)
    return


if __name__ == "__main__":
    main()
