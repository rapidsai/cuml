#
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
import sys
import re
import os
import subprocess
import argparse
import io
from functools import reduce
import operator


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
    argparser.add_argument("--top_include_dirs", required=False,
                           default='src,src_prims',
                           help="comma-separated list of directories used as "
                           "search dirs on build and which should not be "
                           "crossed in relative includes")
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

def rel_include_warnings(dir, src, line_num, inc_file, top_inc_dirs):
    warn = []
    inc_folders = inc_file.split(os.path.sep)[:-1]
    inc_folders_alt = inc_file.split(os.path.altsep)[:-1]
    if len(inc_folders) != 0 and len(inc_folders_alt) != 0:
        w = "File: %s, line %d, warning: using %s and %s as path separators"
        warn.append(w % (src, line_num, os.path.sep, os.path.altsep))
    if len(inc_folders) == 0:
        inc_folders = inc_folders_alt
    abs_inc_folders = [
        os.path.abspath(os.path.join(dir, *inc_folders[:i + 1]))
        for i in range(len(inc_folders))]
    if os.path.curdir in inc_folders:
        w = ("File: %s, line %d, warning: rel include containing "
             "reference to current folder '%s'")
        warn.append(w % (src, line_num, os.path.curdir))
    if any(reduce(operator.or_,
                  [os.path.basename(p) == f for f in top_inc_dirs])
           for p in abs_inc_folders):
        w = "File: %s, line %d, warning: rel include going over %s folders"
        warn.append(w % (src, line_num,
                         "/".join("'" + f + "'" for f in top_inc_dirs)))
    if (len(inc_folders) >= 3 and os.path.pardir in inc_folders and
            any(p != os.path.pardir for p in inc_folders)):
        w = ("File: %s, line %d, warning: rel include with more than "
             "2 folders that aren't in a straight heritage line")
        warn.append(w % (src, line_num))
    return warn

def check_includes_in(src, inplace, top_inc_dirs):
    errs = []
    dir = os.path.dirname(src)
    with io.open(src, encoding="utf-8") as file_obj:
        lines = list(enumerate(file_obj))
    for line_number, line in lines:
        match = IncludeRegex.search(line)
        if match is None:
            continue
        val_type = match.group(2)  # " or <
        inc_file = match.group(3)
        full_path = os.path.join(dir, inc_file)
        line_num = line_number + 1
        if val_type == "\"" and not os.path.isfile(full_path):
            if inplace:
                new_line, n = IncludeRegex.subn(r"\1<\3>", line)
                assert n == 1, "inplace only handles one include match per line"
                errs.append((line_number, new_line))
            else:
                errs.append("Line:%d use #include <...>" % line_num)
        elif val_type == "<" and os.path.isfile(full_path):
            if inplace:
                new_line, n = IncludeRegex.subn(r'\1"\3"', line)
                assert n == 1, "inplace only handles one include match per line"
                errs.append((line_number, new_line))
            else:
                errs.append("Line:%d use #include \"...\"" % line_num)

        # output warnings for some cases
        # 1. relative include containing current folder
        # 2. relative include going over src / src_prims folders
        # 3. relative include longer than 2 folders and containing
        #    both ".." and "non-.."
        # 4. absolute include used but rel include possible without warning
        if val_type == "\"":
            warn = rel_include_warnings(dir, src, line_num, inc_file,
                                        top_inc_dirs)
            for w in warn:
                print(w)
        if val_type == "<":
            # try to make a relative import using the top folders
            for top_folder in top_inc_dirs:
                full_dir = os.path.abspath(dir)
                fs = full_dir.split(os.path.sep)
                fs_alt = full_dir.split(os.path.altsep)
                if len(fs) <= 1:
                    fs = fs_alt
                if top_folder not in fs:
                    continue
                if fs[0] == "":  # full dir was absolute
                    fs[0] = os.path.sep
                full_top = os.path.join(*fs[:fs.index(top_folder) + 1])
                full_inc = os.path.join(full_top, inc_file)
                if not os.path.isfile(full_inc):
                    continue
                new_rel_inc = os.path.relpath(full_inc, full_dir)
                warn = rel_include_warnings(
                    dir, src, line_num, new_rel_inc, top_inc_dirs)
                if len(warn) == 0:
                    w = ("File: %s, line %d, info: absolute include could "
                         "be transformed to relative \"%s\"")
                    print(w % (src, line_num, new_rel_inc))

    if inplace and len(errs) > 0:
        print("File: {}. Changing lines {}".format(
            src, ', '.join(str(x[0]) for x in errs)))
        for line_number, replacement in errs:
            lines[line_number] = (line_number, replacement)
        with io.open(src, "w", encoding="utf-8") as out_file:
            for _, new_line in lines:
                out_file.write(new_line)
        errs = []

    return errs


def main():
    args = parse_args()
    top_inc_dirs = args.top_include_dirs.split(',')
    all_files = list_all_source_file(args.regex_compiled, args.dirs)
    all_errs = {}
    for f in all_files:
        errs = check_includes_in(f, args.inplace, top_inc_dirs)
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
        path_parts = os.path.abspath(__file__).split(os.sep)
        print("You can run '{} --inplace' to bulk fix these errors".format(
            os.sep.join(path_parts[path_parts.index("cpp"):])))
        sys.exit(-1)
    return


if __name__ == "__main__":
    main()
