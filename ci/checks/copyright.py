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

import datetime
import re
import gitutils
import argparse
import io
import os


FilesToCheck = [
    re.compile(r"[.](cmake|cpp|cu|cuh|h|hpp|sh|pxd|py|pyx)$"),
    re.compile(r"CMakeLists[.]txt$"),
    re.compile(r"CMakeLists_standalone[.]txt$"),
    re.compile(r"setup[.]cfg$"),
    re.compile(r"meta[.]yaml$")
]

# this will break starting at year 10000, which is probably OK :)
CheckSimple = re.compile(r"Copyright \(c\) (\d{4}), NVIDIA CORPORATION")
CheckDouble = re.compile(
    r"Copyright \(c\) (\d{4})-(\d{4}), NVIDIA CORPORATION")


def checkThisFile(f):
    if gitutils.isFileEmpty(f):
        return False
    for checker in FilesToCheck:
        if checker.search(f):
            return True
    return False


def getCopyrightYears(line):
    res = CheckSimple.search(line)
    if res:
        return (int(res.group(1)), int(res.group(1)))
    res = CheckDouble.search(line)
    if res:
        return (int(res.group(1)), int(res.group(2)))
    return (None, None)


def replaceCurrentYear(line, start, end):
    # first turn a simple regex into double (if applicable). then update years
    res = CheckSimple.sub(r"Copyright (c) \1-\1, NVIDIA CORPORATION", line)
    res = CheckDouble.sub(
        r"Copyright (c) {:04d}-{:04d}, NVIDIA CORPORATION".format(start, end),
        res)
    return res


def checkCopyright(f, update_current_year):
    """
    Checks for copyright headers and their years
    """
    errs = []
    thisYear = datetime.datetime.now().year
    lineNum = 0
    crFound = False
    yearMatched = False
    with io.open(f, "r", encoding="utf-8") as fp:
        lines = fp.readlines()
    for line in lines:
        lineNum += 1
        start, end = getCopyrightYears(line)
        if start is None:
            continue
        crFound = True
        if start > end:
            e = [f, lineNum, "First year after second year in the copyright "
                 "header (manual fix required)", None]
            errs.append(e)
        if thisYear < start or thisYear > end:
            e = [f, lineNum, "Current year not included in the "
                 "copyright header", None]
            if thisYear < start:
                e[-1] = replaceCurrentYear(line, thisYear, end)
            if thisYear > end:
                e[-1] = replaceCurrentYear(line, start, thisYear)
            errs.append(e)
        else:
            yearMatched = True
    fp.close()
    # copyright header itself not found
    if not crFound:
        e = [f, 0, "Copyright header missing or formatted incorrectly "
             "(manual fix required)", None]
        errs.append(e)
    # even if the year matches a copyright header, make the check pass
    if yearMatched:
        errs = []

    if update_current_year:
        errs_update = [x for x in errs if x[-1] is not None]
        if len(errs_update) > 0:
            print("File: {}. Changing line(s) {}".format(
                f, ', '.join(str(x[1]) for x in errs if x[-1] is not None)))
            for _, lineNum, __, replacement in errs_update:
                lines[lineNum - 1] = replacement
            with io.open(f, "w", encoding="utf-8") as out_file:
                for new_line in lines:
                    out_file.write(new_line)
        errs = [x for x in errs if x[-1] is None]

    return errs


def checkCopyrightForAll():
    """
    Checks for copyright headers in all the modified files. In case of local
    repo, this script will just look for uncommitted files and in case of CI
    it compares between branches "$PR_TARGET_BRANCH" and "current-pr-branch"
    """
    argparser = argparse.ArgumentParser(
        "Checks for a consistent copyright header in git's modified files")
    argparser.add_argument("--update-current-year", dest='update_current_year',
                           action="store_true", required=False, help="If set, "
                           "update the current year if a header is already "
                           "present and well formatted.")
    args = argparser.parse_args()
    files = gitutils.modifiedFiles(filter=checkThisFile)
    errors = []
    for f in files:
        errors += checkCopyright(f, args.update_current_year)
    if len(errors) > 0:
        print("Copyright headers incomplete in some of the files!")
        for e in errors:
            print("  %s:%d Issue: %s" % (e[0], e[1], e[2]))
        print("")
        n_fixable = sum(1 for e in errors if e[-1] is not None)
        path_parts = os.path.abspath(__file__).split(os.sep)
        file_from_cuml = os.sep.join(path_parts[path_parts.index("ci"):])
        if n_fixable > 0:
            print("You can run {} --update-current-year to fix {} of these "
                  "errors.\n".format(file_from_cuml, n_fixable))
        raise Exception("Copyright check failed! Check above to know more.")
    else:
        print("Copyright check passed")


if __name__ == "__main__":
    checkCopyrightForAll()
