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

import datetime
import re
import gitutils


FilesToCheck = [
    re.compile(r"[.](cmake|cpp|cu|cuh|h|hpp|sh|pxd|py|pyx)$"),
    re.compile(r"CMakeLists[.]txt$"),
    re.compile(r"CMakeLists_standalone[.]txt$"),
    re.compile(r"setup[.]cfg$"),
    re.compile(r"[.]flake8[.]cython$"),
    re.compile(r"meta[.]yaml$")
]


def checkThisFile(f):
    if gitutils.isFileEmpty(f):
        return False
    for checker in FilesToCheck:
        if checker.search(f):
            return True
    return False


def getCopyrightYears(line):
    res = re.search(r"Copyright \(c\) (\d{4}), NVIDIA CORPORATION", line)
    if res:
        return (int(res.group(1)), int(res.group(1)))
    res = re.search(r"Copyright \(c\) (\d{4})-(\d{4}), NVIDIA CORPORATION",
                    line)
    if res:
        return (int(res.group(1)), int(res.group(2)))
    return (None, None)


def checkCopyright(f):
    """
    Checks for copyright headers and their years
    """
    errs = []
    thisYear = datetime.datetime.now().year
    lineNum = 0
    crFound = False
    yearMatched = False
    fp = open(f, "r")
    for line in fp.readlines():
        lineNum += 1
        start, end = getCopyrightYears(line)
        if start is None:
            continue
        crFound = True
        if thisYear < start or thisYear > end:
            errs.append((f, lineNum,
                         "Current year not included in the copyright header"))
        else:
            yearMatched = True
    fp.close()
    # copyright header itself not found
    if not crFound:
        errs.append((f, 0,
                     "Copyright header missing or formatted incorrectly"))
    # even if the year matches a copyright header, make the check pass
    if yearMatched:
        errs = []
    return errs


def checkCopyrightForAll():
    """
    Checks for copyright headers in all the modified files. In case of local
    repo, this script will just look for uncommitted files and in case of CI
    it compares between branches "$PR_TARGET_BRANCH" and "current-pr-branch"
    """
    files = gitutils.modifiedFiles(filter=checkThisFile)
    errors = []
    for f in files:
        errors += checkCopyright(f)
    if len(errors) > 0:
        print("Copyright headers incomplete in some of the files!")
        for e in errors:
            print("  %s:%d Issue: %s" % (e[0], e[1], e[2]))
        print("")
        raise Exception("Copyright check failed! Check above to know more")
    else:
        print("Copyright check passed")


if __name__ == "__main__":
    checkCopyrightForAll()
