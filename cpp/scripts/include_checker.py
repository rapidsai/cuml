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


# file names could (in theory) contain simple white-space
IncludeRegex = re.compile(r"(\s*#include\s*)([\"<])([\S ]+)([\">])")


def parse_args():
    argparser = argparse.ArgumentParser(
        "Checks for a consistent '#include' syntax")
    argparser.add_argument("--regex", type=str,
                           default=r"[.](cu|cuh|h|hpp|hxx|cpp)$",
                           help="Regex string to filter in sources")
    argparser.add_argument("--inplace", action="store_true", required=False,
                           help="If set, perform the required changes inplace.")
    argparser.add_argument("dirs", type=str, nargs="*",
                           help="List of dirs where to find sources")
    args = argparser.parse_args()
    args.regex_compiled = re.compile(args.regex)
    return args


def list_all_source_file(file_regex, srcdirs):
    all_files = []
    for srcdir in srcdirs:
        for root, dirs, files in os.walk(srcdir):
            for f in files:
                if re.search(file_regex, f):
                    src = os.path.join(root, f)
                    all_files.append(src)
    return all_files


def check_includes_in(src, inplace):
    errs = []
    dir = os.path.dirname(src)
    with open(src) as file_obj:
        lines = list(enumerate(file_obj))
    for line_number, line in lines:
        match = IncludeRegex.search(line)
        if match is None:
            continue
        val_type = match.group(2)  # " or <
        inc_file = match.group(3)
        full_path = os.path.join(dir, inc_file)
        line_num = line_number + 1
        if val_type == "\"" and not os.path.exists(full_path):
            if inplace:
                new_line, n = IncludeRegex.subn(r"\1<\3>", line)
                assert n == 1, "inplace only handles one include match per line"
                errs.append((line_number, new_line))
            else:
                errs.append("Line:%d use #include <...>" % line_num)
        elif val_type == "<" and os.path.exists(full_path):
            if inplace:
                new_line, n = IncludeRegex.subn(r'\1"\3"', line)
                assert n == 1, "inplace only handles one include match per line"
                errs.append((line_number, new_line))
            else:
                errs.append("Line:%d use #include \"...\"" % line_num)

    if inplace and len(errs) > 0:
        print("File: {}. Changing lines {}".format(
            src, ', '.join(str(x[0]) for x in errs)))
        for line_number, replacement in errs:
            lines[line_number] = (line_number, replacement)
        with open(src, 'w') as out_file:
            for _, new_line in lines:
                out_file.write(new_line)
        errs = []

    return errs


def main():
    args = parse_args()
    all_files = list_all_source_file(args.regex_compiled, args.dirs)
    all_errs = {}
    for f in all_files:
        errs = check_includes_in(f, args.inplace)
        if len(errs) > 0:
            all_errs[f] = errs
    if len(all_errs) == 0:
        print("include-check PASSED")
    else:
        print("include-check FAILED! See below for errors...")
        for f, errs in all_errs.items():
            print("File: %s" % f)
            for e in errs:
                print("  %s" % e)
        sys.exit(-1)
    return


if __name__ == "__main__":
    main()
