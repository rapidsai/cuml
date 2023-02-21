# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

import argparse
import datetime
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

import git

LOGGER = logging.getLogger("copyright.py")

FILES_TO_INCLUDE = [
    re.compile(r"[.](cmake|cpp|cu|cuh|h|hpp|sh|pxd|py|pyx)$"),
    re.compile(r"CMakeLists[.]txt$"),
    re.compile(r"CMakeLists_standalone[.]txt$"),
    re.compile(r"setup[.]cfg$"),
    re.compile(r"[.]flake8[.]cython$"),
    re.compile(r"meta[.]yaml$"),
]
FILES_TO_EXCLUDE = [
    re.compile(r"cpp/src/tsne/cannylab/bh\.cu"),
]

# this will break starting at year 10000, which is probably OK :)
RE_CHECK_SIMPLE = re.compile(
    r"Copyright *(?:\(c\))? *(\d{4}),? *NVIDIA C(?:ORPORATION|orporation)"
)
RE_CHECK_DOUBLE = re.compile(
    r"Copyright *(?:\(c\))? *(\d{4})-(\d{4}),? *NVIDIA C(?:ORPORATION|orporation)"  # noqa: E501
)


class _GitDiffPath(os.PathLike):
    """Utility class to provide PathLike interface for git diff blobs."""

    def __init__(self, diff):
        self._diff = diff

    def __fspath__(self):
        return self._diff.b_path

    def __str__(self):
        return str(self.__fspath__())

    def exists(self):
        return self._diff.b_path is not None and self._diff.b_blob.size > 0

    def is_file(self):
        return self.exists()

    def read_text(self, encoding="utf-8", errors="strict"):
        return self._diff.b_blob.data_stream.read().decode(
            encoding=encoding, errors=errors
        )

    def write_text(self, data, encoding=None, errors=None, newline=None):
        Path(self).write_text(
            data, encoding=encoding, errors=errors, newline=newline
        )


def find_upstream_base_branch(base_branch: str):
    if base_branch in repo.refs:
        # Use the tracking branch of the local reference if it exists. This
        # returns None if no tracking branch is set.
        return repo.refs[base_branch]

    else:
        candidate_branches = [
            remote.refs[base_branch]
            for remote in repo.remotes
            if base_branch in remote.refs
        ]
        if len(candidate_branches) > 0:
            return sorted(
                candidate_branches,
                key=lambda branch: branch.commit.committed_datetime,
            )[-1]

    raise RuntimeError(f"Unable to find ref for '{base_branch}'.")


def find_git_modified_files(repo, upstream_target_branch):
    """Get a set of all modified files, as Diff objects.

    The files returned have been modified in git since the merge base of HEAD
    and the upstream of the target branch. We return the Diff objects so that
    we can read only the staged changes.
    """

    merge_base = repo.merge_base("HEAD", upstream_target_branch.commit)[0]
    for f in merge_base.diff():
        if f.b_path is not None and not f.deleted_file and f.b_blob.size > 0:
            yield _GitDiffPath(f)


def get_copyright_years(line):
    if res := RE_CHECK_SIMPLE.search(line):
        return int(res.group(1)), int(res.group(1))
    if res := RE_CHECK_DOUBLE.search(line):
        return int(res.group(1)), int(res.group(2))
    return None, None


def replace_current_year(line, start, end):
    # first turn a simple regex into double (if applicable). then update years
    res = RE_CHECK_SIMPLE.sub(r"Copyright (c) \1-\1, NVIDIA CORPORATION", line)
    res = RE_CHECK_DOUBLE.sub(
        rf"Copyright (c) {start:04d}-{end:04d}, NVIDIA CORPORATION",
        res,
    )
    return res


def find_copyright_issues(path: Path):
    """Checks for copyright headers and their years."""
    if not path.is_file():
        yield 0, f"No such file or a directory '{path}'.", None
        return

    this_year = datetime.datetime.now().year

    lines = path.read_text().splitlines()

    for line_no, line in enumerate(lines):
        start, end = get_copyright_years(line)
        if start is None:
            continue
        if start > end:
            yield line_no, "First year after second year in the copyright header (manual fix required)", None
        elif this_year < start:
            yield line_no, "Current year not included in the copyright header", replace_current_year(
                line, this_year, end
            )
        elif this_year > end:
            yield line_no, "Current year not included in the copyright header", replace_current_year(
                line, start, this_year
            )
        break
    else:
        # Did not find copyright header.
        yield 0, "Copyright header missing or formatted incorrectly (manual fix required)", None


def perform_updates(path: Path, updates):
    lines = path.read_text().split("\n")
    for line_no, fix in updates:
        if fix:
            lines[line_no] = fix
            print(f"Fixed {path}:{line_no}")
    path.write_text("\n".join(lines))


def find_files(paths, repo, target_branch):
    if paths and target_branch:
        git_modified = [
            os.fspath(f) for f in find_git_modified_files(repo, target_branch)
        ]
        for path in paths:
            yield from (f for f in walk(path) if os.fspath(f) in git_modified)
    elif paths:
        for path in paths:
            yield from walk(path)
    elif target_branch:
        yield from find_git_modified_files(repo, target_branch)
    else:
        yield from walk(Path.cwd())


def walk(top: Optional[os.PathLike], pathFilter=None):

    if top.is_file():
        if pathFilter is None or pathFilter(top):
            yield top
    elif top.is_dir():
        for root, _, files in os.walk(top):
            for name in files:
                path = Path(top, root, name)
                if pathFilter is None or pathFilter(path):
                    yield path
    else:
        yield top


def main(repo):
    """
    Checks for copyright headers in all the modified files. In case of local
    repo, this script will just look for uncommitted files and in case of CI
    it compares between branches "$PR_TARGET_BRANCH" and "current-pr-branch"
    """

    default_base_branch = str(repo.active_branch)

    argparser = argparse.ArgumentParser(
        "Checks for a consistent copyright header in git's modified files"
    )
    argparser.add_argument(
        dest="files",
        metavar="filename",
        nargs="*",
        type=Path,
        help="If provided, only check the files the provided names.",
    )
    argparser.add_argument(
        "-i",
        "--fix-in-place",
        action="store_true",
        required=False,
        help="If set, "
        "update the current year if a header is already "
        "present and well formatted.",
    )
    argparser.add_argument(
        "--base-branch",
        default=default_base_branch,
        help=f"Specify which branch/commit to compare against (default: {default_base_branch})",
    )
    argparser.add_argument(
        "--exclude",
        dest="exclude",
        action="append",
        default=["python/cuml/_thirdparty/"],
        help=(
            "Exclude the paths specified (regexp). "
            "Can be specified multiple times."
        ),
    )
    argparser.add_argument("-v", "--verbosity", action="count", default=0)

    args = argparser.parse_args()

    logging.basicConfig(level=max(0, logging.ERROR - args.verbosity * 10))

    upstream_base_branch = find_upstream_base_branch(str(args.base_branch))

    exempted = FILES_TO_EXCLUDE + [re.compile(ex) for ex in args.exclude]

    def include(path: os.PathLike):
        for exempt in exempted:
            if exempt.search(os.fspath(path)):
                return False
        for included in FILES_TO_INCLUDE:
            if included.search(os.fspath(path)):
                break
        else:
            return False
        return path.is_file()

    files = [
        f
        for f in find_files(args.files, repo, upstream_base_branch)
        if include(f)
    ]

    issues = [(file, list(find_copyright_issues(file))) for file in files]

    num_fixes = 0
    num_issues = 0

    for path, issues_ in issues:
        fixes = []
        for line_no, description, fix in issues_:
            print(f"{path}:{line_no}: {description}")
            num_issues += 1
            if fix:
                fixes.append((line_no, fix))
        num_fixes += len(fixes)
        if fixes and args.fix_in_place:
            perform_updates(path, fixes)

    try:
        if args.fix_in_place:
            if num_fixes < num_issues:
                raise RuntimeError("Unable to fix all issues.")
        elif 0 < num_fixes < num_issues:
            raise RuntimeError(
                "Use the '-i/--fix-in-place' option to fix some of the issues."
            )
        elif 0 < num_fixes == num_issues:
            raise RuntimeError(
                "Use the '-i/--fix-in-place' option to fix the issues."
            )
    except RuntimeError as message:
        print(message)
        return 1
    else:
        return 0


if __name__ == "__main__":
    repo = git.Repo()
    sys.exit(main(repo))
