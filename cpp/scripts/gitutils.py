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

import subprocess
import os


def isFileEmpty(f):
    return os.stat(f).st_size == 0


def __git(*opts):
    """Runs a git command and returns its output"""
    cmd = "git " + " ".join(list(opts))
    return subprocess.check_output(cmd, shell=True)


def __gitdiff(*opts):
    """Runs a git diff command with no pager set"""
    return __git("--no-pager", "diff", *opts)


def branch():
    """Returns the name of the current branch"""
    name = __git("rev-parse", "--abbrev-ref", "HEAD")
    name = name.rstrip()
    return name


def uncommittedFiles():
    """
    Returns a list of all changed files that are not yet committed. This
    means both untracked/unstaged as well as uncommitted files too.
    """
    files = __gitdiff("--name-only", "--cached", "--ignore-submodules")
    ret = files.splitlines()
    files = __git("status", "-u", "-s")
    for f in files.splitlines():
        f = f.decode(encoding='UTF-8')
        f = f.strip(" ")
        tmp = f.split(" ", 1)
        # only consider staged files or uncommitted files
        # in other words, ignore untracked files
        if tmp[0] == "M" or tmp[0] == "A":
            ret.append(tmp[1])
    return ret


def changedFilesBetween(b1, b2):
    """Returns a list of files changed between branches b1 and b2"""
    current = branch()
    __git("checkout", "--quiet", b1)
    __git("checkout", "--quiet", b2)
    files = __gitdiff("--name-only", "--ignore-submodules", "%s...%s" %
                      (b1, b2))
    __git("checkout", "--quiet", current)
    return files.splitlines()


def changesInFileBetween(file, b1, b2, filter=None):
    """Filters the changed lines to a file between the branches b1 and b2"""
    current = branch()
    __git("checkout", "--quiet", b1)
    __git("checkout", "--quiet", b2)
    diffs = __gitdiff("--ignore-submodules", "-w", "--minimal", "-U0",
                      "%s...%s" % (b1, b2), "--", file)
    __git("checkout", "--quiet", current)
    lines = []
    for line in diffs.splitlines():
        if filter is None or filter(line):
            lines.append(line)
    return lines


def modifiedFiles(filter=None):
    """
    If inside a CI-env (ie. currentBranch=current-pr-branch and the env-var
    PR_TARGET_BRANCH is defined), then lists out all files modified between
    these 2 branches. Else, lists out all the uncommitted files in the current
    branch.

    Such utility function is helpful while putting checker scripts as part of
    cmake, as well as CI process. This way, during development, only the files
    touched (but not yet committed) by devs can be checked. But, during the CI
    process ALL files modified by the dev, as submiited in the PR, will be
    checked. This happens, all the while using the same script.
    """
    if "PR_TARGET_BRANCH" in os.environ and branch() == "current-pr-branch":
        allFiles = changedFilesBetween(os.environ["PR_TARGET_BRANCH"],
                                       branch())
    else:
        allFiles = uncommittedFiles()
    files = []
    for f in allFiles:
        if filter is None or filter(f):
            files.append(f)
    return files


def listAllFilesInDir(folder):
    """Utility function to list all files/subdirs in the input folder"""
    allFiles = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            allFiles.append(os.path.join(root, name))
    return allFiles


def listFilesToCheck(filesDirs, filter=None):
    """
    Utility function to filter the input list of files/dirs based on the input
    filter method and returns all the files that need to be checked
    """
    allFiles = []
    for f in filesDirs:
        if os.path.isfile(f):
            if filter is None or filter(f):
                allFiles.append(f)
        elif os.path.isdir(f):
            files = listAllFilesInDir(f)
            for f in files:
                if filter is None or filter(f):
                    allFiles.append(f)
    return allFiles
