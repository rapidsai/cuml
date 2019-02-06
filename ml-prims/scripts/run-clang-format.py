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


# TODO: prepare 'dst' correctly!
def listAllSources(fileRegexStr, srcdir, bindir, inplace):
    fileRegex = re.compile(fileRegexStr)
    allFiles = []
    for d, s, files in os.walk(srcdir):
        for f in files:
            if re.search(fileRegex, f):
                src = os.path.join(d, f)
                dst = src + ".clang.format" if not inplace else src
                allFiles.append((src, dst))
    return allFiles

def parseArgs():
    argparser = argparse.ArgumentParser("Run clang-format on a project")
    argparser.add_argument("-bindir", type=str, default=".",
                           help="Path to the current build directory.")
    argparser.add_argument("-exe", type=str, default="clang-format",
                           help="Path to clang-format exe")
    argparser.add_argument("-inplace", default=False, action="store_true",
                           help="Replace the source files itself.")
    argparser.add_argument("-regex", type=str, default=r"[.](h|cpp|cu)$",
                           help="Regex string to filter in sources")
    argparser.add_argument("-srcdir", type=str, default=".",
                           help="Path to directory containing the dirs.")
    argparser.add_argument("dirs", type=str, nargs="*",
                           help="List of dirs where to find sources")
    return argparser.parse_args()

def isNewer(src, dst):
    if not os.path.exists(dst):
        return True
    a = os.path.getmtime(src)
    b = os.path.getmtime(dst)
    return a >= b

def runClangFormat(src, dst, exe):
    # run the clang format command itself
    if isNewer(src, dst):
        if src == dst:
            cmd = "%s -i %s" % (exe, src)
        else:
            cmd = "%s %s > %s" % (exe, src, dst)
        try:
            subprocess.check_call(cmd, shell=True)
        except subprocess.CalledProcessError:
            print("Unable to run clang-format! Please configure your environment.")
            raise
    # run the diff to check if there are any formatting issues
    cmd = "diff -q %s %s >/dev/null" % (src, dst)
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError:
        print("clang-format failed! Run 'diff %s %s' to see the formatting issues!" %
              (src, dst))
        return False
    return True

def main():
    args = parseArgs()
    allFiles = []
    for d in args.dirs:
        srcdir = os.path.join(args.srcdir, d)
        bindir = os.path.join(args.bindir, d)
        allFiles += listAllSources(args.regex, srcdir, bindir, args.inplace)
    status = True
    for src, dst in allFiles:
        if not runClangFormat(src, dst, args.exe):
            status = False
    if not status:
        print("Clang-format failed! Look at errors above and fix them, or if")
        print("you trust the clang-format, you can also run the following")
        print("command to bulk fix the files!")
        print("  %s -inplace -srcdir %s -exe %s %s" %
              (sys.argv[0], args.srcdir, args.exe, " ".join(args.dirs)))
        sys.exit(-1)
    return

if __name__ == "__main__":
    main()
