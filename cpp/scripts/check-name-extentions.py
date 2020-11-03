# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
import re
import os
import sys
import subprocess
import argparse



DEFAULT_DIRS = ["cpp/bench",
                "cpp/comms/mpi/include",
                "cpp/comms/mpi/src",
                "cpp/comms/std/include",
                "cpp/comms/std/src",
                "cpp/include",
                "cpp/examples",
                "cpp/src",
                "cpp/src_prims",
                "cpp/test"]

IGNORE_FILES = ""
Search_Regex = re.compile(r"(#include.*.cuh|__device__|__global__)")

def parse_args():
    argparser = argparse.ArgumentParser("Runs clang-format on a project")
    argparser.add_argument("-regex", type=str,
                           default=r"[.](h|hpp|cpp)$",
                           help="Regex string to filter in sources")
    argparser.add_argument("-ignore", type=str, default=IGNORE_FILES,
                           help="Regex used to ignore files from matched list")
    argparser.add_argument("dirs", type=str, nargs="*",
                           help="List of dirs where to find sources")
    args = argparser.parse_args()
    args.regex_compiled = re.compile(args.regex)
    args.ignore_compiled = re.compile(args.ignore)
    if len(args.dirs) == 0:
        args.dirs = DEFAULT_DIRS
    return args

def list_all_src_files(file_regex, ignore_regex, srcdirs):
    allFiles = []
    for srcdir in srcdirs:
        for root, dirs, files in os.walk(srcdir):
            for f in files:
                if re.search(file_regex, f):
                    src = os.path.join(root, f)
                    if re.search(ignore_regex, src):
                        continue
                    allFiles.append(src)
    return allFiles

def test_file(filename):
    number_of_conflicts_in_file = 0
    file = open(filename, "r")
    for line in file:
        if Search_Regex.search(line):
            print('Naming Conflict in : ' + filename + ':' + line)
            number_of_conflicts_in_file += 1
    return number_of_conflicts_in_file

def main():
    args = parse_args()
    if not os.path.exists(".git"):
        print("Error!! This needs to always be run from the root of repo")
        sys.exit(-1)
    all_files = list_all_src_files(args.regex_compiled, args.ignore_compiled,
                                   args.dirs)
    num_conflicts = 0
    for file in all_files:
        num_conflicts += test_file(file)
    print('Total Number of Naming conflicts : ' + str(num_conflicts))
    if num_conflicts > 0:
        print('Please refer to #1675 to the naming rules of files and includes')
        sys.exit(-1)


if __name__ == "__main__":
    main()

