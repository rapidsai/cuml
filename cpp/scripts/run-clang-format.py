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
import gitutils


def listAllSourceFiles(fileRegex, srcdirs, dstdir, inplace):
    allFiles = []
    for srcdir in srcdirs:
        for root, dirs, files in os.walk(srcdir):
            for f in files:
                if re.search(fileRegex, f):
                    src = os.path.join(root, f)
                    if inplace:
                        _dir = root
                    else:
                        _dir = os.path.join(dstdir, os.path.basename(root))
                    dst = os.path.join(_dir, f)
                    allFiles.append((src, dst))
    return allFiles


def listAllChangedFiles(fileRegex, dstdir, inplace):
    allFiles = []

    def checkThisFile(f):
        return True if fileRegex.search(f) else False

    files = gitutils.modifiedFiles(filter=checkThisFile)
    for f in files:
        dst = f if inplace else os.path.join(dstdir, f)
        allFiles.append((f, dst))
    return allFiles


def parseArgs():
    argparser = argparse.ArgumentParser("Runs clang-format on a project")
    argparser.add_argument("-dstdir", type=str, default=".",
                           help="Path to the current build directory.")
    argparser.add_argument("-exe", type=str, default="clang-format",
                           help="Path to clang-format exe")
    argparser.add_argument("-inplace", default=False, action="store_true",
                           help="Replace the source files itself.")
    argparser.add_argument("-regex", type=str,
                           default=r"[.](cu|cuh|h|hpp|cpp)$",
                           help="Regex string to filter in sources")
    argparser.add_argument("-onlyChangedFiles", default=False,
                           action="store_true",
                           help="Run the formatter on changedFiles only.")
    argparser.add_argument("-srcdir", type=str, default=".",
                           help="Path to directory containing the dirs.")
    argparser.add_argument("dirs", type=str, nargs="*",
                           help="List of dirs where to find sources")
    args = argparser.parse_args()
    args.regexCompiled = re.compile(args.regex)
    return args


def isNewer(src, dst):
    if not os.path.exists(dst):
        return True
    a = os.path.getmtime(src)
    b = os.path.getmtime(dst)
    return a >= b


def runClangFormat(src, dst, exe):
    dstdir = os.path.dirname(dst)
    if not os.path.exists(dstdir):
        os.makedirs(dstdir)
    # run the clang format command itself
    if isNewer(src, dst):
        if src == dst:
            cmd = "%s -i %s" % (exe, src)
        else:
            cmd = "%s %s > %s" % (exe, src, dst)
        try:
            subprocess.check_call(cmd, shell=True)
        except subprocess.CalledProcessError:
            print("Unable to run clang-format! Maybe your env is not "
                  "configured properly?")
            raise
    # run the diff to check if there are any formatting issues
    cmd = "diff -q %s %s >/dev/null" % (src, dst)
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError:
        print("clang-format failed! Run 'diff %s %s' to know more about the "
              "formatting violations!" %
              (src, dst))
        return False
    return True


def main():
    args = parseArgs()
    # Either run the format checker on only the changed files or all the source
    # files as specified by the user
    if args.onlyChangedFiles:
        allFiles = listAllChangedFiles(args.regexCompiled, args.dstdir,
                                       args.inplace)
    else:
        allFiles = listAllSourceFiles(args.regexCompiled, args.dirs,
                                      args.dstdir, args.inplace)
    # actual format checker
    status = True
    for src, dst in allFiles:
        if not runClangFormat(src, dst, args.exe):
            status = False
    if not status:
        print("Clang-format failed! Look at errors above and fix them, or if")
        print("you trust the clang-format, you can also run the following")
        print("command to bulk fix the files!")
        if args.onlyChangedFiles:
            print("  python %s -inplace -onlyChangedFiles -dstdir %s -exe %s" %
                  (sys.argv[0], args.dstdir, args.exe))
        else:
            print("  python %s -inplace -dstdir %s -srcdir %s -exe %s %s" %
                  (sys.argv[0], args.dstdir, args.srcdir, args.exe,
                   " ".join(args.dirs)))
        sys.exit(-1)
    return


if __name__ == "__main__":
    main()
