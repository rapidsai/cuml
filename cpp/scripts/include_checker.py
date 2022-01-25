#
# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
import argparse
import io
from functools import reduce
import operator
import dataclasses
import typing

# file names could (in theory) contain simple white-space
IncludeRegex = re.compile(r"(\s*#include\s*)([\"<])([\S ]+)([\">])")
PragmaRegex = re.compile(r"^ *\#pragma\s+once *$")


def parse_args():
    argparser = argparse.ArgumentParser(
        "Checks for a consistent '#include' syntax")
    argparser.add_argument("--regex",
                           type=str,
                           default=r"[.](cu|cuh|h|hpp|hxx|cpp)$",
                           help="Regex string to filter in sources")
    argparser.add_argument(
        "--inplace",
        action="store_true",
        required=False,
        help="If set, perform the required changes inplace.")
    argparser.add_argument("--top_include_dirs",
                           required=False,
                           default='src,src_prims',
                           help="comma-separated list of directories used as "
                           "search dirs on build and which should not be "
                           "crossed in relative includes")
    argparser.add_argument("dirs",
                           type=str,
                           nargs="*",
                           help="List of dirs where to find sources")
    args = argparser.parse_args()
    args.regex_compiled = re.compile(args.regex)
    return args


@dataclasses.dataclass()
class Issue:
    is_error: bool
    msg: str
    file: str
    line: int
    fixed_str: str = None
    was_fixed: bool = False

    def get_msg_str(self) -> str:
        if (self.is_error and not self.was_fixed):
            return make_error_msg(
                self.file,
                self.line,
                self.msg + (". Fixed!" if self.was_fixed else ""))
        else:
            return make_warn_msg(
                self.file,
                self.line,
                self.msg + (". Fixed!" if self.was_fixed else ""))


def make_msg(err_or_warn: str, file: str, line: int, msg: str):
    """
    Formats the error message with a file and line number that can be used by
    IDEs to quickly go to the exact line
    """
    return "{}: {}:{}, {}".format(err_or_warn, file, line, msg)


def make_error_msg(file: str, line: int, msg: str):
    return make_msg("ERROR", file, line, msg)


def make_warn_msg(file: str, line: int, msg: str):
    return make_msg("WARN", file, line, msg)


def list_all_source_file(file_regex, srcdirs):
    all_files = []
    for srcdir in srcdirs:
        for root, dirs, files in os.walk(srcdir):
            for f in files:
                if re.search(file_regex, f):
                    src = os.path.join(root, f)
                    all_files.append(src)
    return all_files


def rel_include_warnings(dir, src, line_num, inc_file,
                         top_inc_dirs) -> typing.List[Issue]:
    warn: typing.List[Issue] = []
    inc_folders = inc_file.split(os.path.sep)[:-1]
    inc_folders_alt = inc_file.split(os.path.altsep)[:-1]

    if len(inc_folders) != 0 and len(inc_folders_alt) != 0:
        w = "using %s and %s as path separators" % (os.path.sep,
                                                    os.path.altsep)
        warn.append(Issue(False, w, src, line_num))

    if len(inc_folders) == 0:
        inc_folders = inc_folders_alt

    abs_inc_folders = [
        os.path.abspath(os.path.join(dir, *inc_folders[:i + 1]))
        for i in range(len(inc_folders))
    ]

    if os.path.curdir in inc_folders:
        w = "rel include containing reference to current folder '{}'".format(
            os.path.curdir)
        warn.append(Issue(False, w, src, line_num))

    if any(
            any([os.path.basename(p) == f for f in top_inc_dirs])
            for p in abs_inc_folders):

        w = "rel include going over %s folders" % ("/".join(
            "'" + f + "'" for f in top_inc_dirs))

        warn.append(Issue(False, w, src, line_num))

    if (len(inc_folders) >= 3 and os.path.pardir in inc_folders
            and any(p != os.path.pardir for p in inc_folders)):

        w = ("rel include with more than "
             "2 folders that aren't in a straight heritage line")
        warn.append(Issue(False, w, src, line_num))

    return warn


def check_includes_in(src, inplace, top_inc_dirs) -> typing.List[Issue]:
    issues: typing.List[Issue] = []
    dir = os.path.dirname(src)
    found_pragma_once = False
    include_count = 0

    # Read all lines
    with io.open(src, encoding="utf-8") as file_obj:
        lines = list(enumerate(file_obj))

    for line_number, line in lines:
        line_num = line_number + 1

        match = IncludeRegex.search(line)
        if match is None:
            # Check to see if its a pragma once
            if not found_pragma_once:
                pragma_match = PragmaRegex.search(line)

                if pragma_match is not None:
                    found_pragma_once = True

                    if include_count > 0:
                        issues.append(
                            Issue(
                                True,
                                "`#pragma once` must be before any `#include`",
                                src,
                                line_num))
            continue

        include_count += 1

        val_type = match.group(2)  # " or <
        inc_file = match.group(3)
        full_path = os.path.join(dir, inc_file)

        if val_type == "\"" and not os.path.isfile(full_path):
            new_line, n = IncludeRegex.subn(r"\1<\3>", line)
            assert n == 1, "inplace only handles one include match per line"

            issues.append(
                Issue(True, "use #include <...>", src, line_num, new_line))

        elif val_type == "<" and os.path.isfile(full_path):
            new_line, n = IncludeRegex.subn(r'\1"\3"', line)
            assert n == 1, "inplace only handles one include match per line"

            issues.append(
                Issue(True, "use #include \"...\"", src, line_num, new_line))

        # output warnings for some cases
        # 1. relative include containing current folder
        # 2. relative include going over src / src_prims folders
        # 3. relative include longer than 2 folders and containing
        #    both ".." and "non-.."
        # 4. absolute include used but rel include possible without warning
        if val_type == "\"":
            issues += rel_include_warnings(dir,
                                           src,
                                           line_num,
                                           inc_file,
                                           top_inc_dirs)
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
                warn = rel_include_warnings(dir,
                                            src,
                                            line_num,
                                            new_rel_inc,
                                            top_inc_dirs)
                if len(warn) == 0:
                    issues.append(
                        Issue(
                            False,
                            "absolute include could be transformed to relative",
                            src,
                            line_num,
                            f"#include \"{new_rel_inc}\"\n"))
                else:
                    issues += warn

    if inplace and len(issues) > 0:
        had_fixes = False

        for issue in issues:
            if (issue.fixed_str is not None):
                lines[issue.line - 1] = (lines[issue.line - 1][0],
                                         issue.fixed_str)
                issue.was_fixed = True
                had_fixes = True

        if (had_fixes):
            with io.open(src, "w", encoding="utf-8") as out_file:
                for _, new_line in lines:
                    out_file.write(new_line)

    return issues


def main():
    args = parse_args()
    top_inc_dirs = args.top_include_dirs.split(',')
    all_files = list_all_source_file(args.regex_compiled, args.dirs)
    all_issues: typing.List[Issue] = []
    errs: typing.List[Issue] = []

    for f in all_files:
        issues = check_includes_in(f, args.inplace, top_inc_dirs)

        all_issues += issues

    for i in all_issues:
        if (i.is_error and not i.was_fixed):
            errs.append(i)
        else:
            print(i.get_msg_str())

    if len(errs) == 0:
        print("include-check PASSED")
    else:
        print("include-check FAILED! See below for errors...")
        for err in errs:
            print(err.get_msg_str())

        path_parts = os.path.abspath(__file__).split(os.sep)
        print("You can run '{} --inplace' to bulk fix these errors".format(
            os.sep.join(path_parts[path_parts.index("cpp"):])))
        sys.exit(-1)
    return


if __name__ == "__main__":
    main()
