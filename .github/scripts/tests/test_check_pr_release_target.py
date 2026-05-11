# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "check_pr_release_target.py"
)
SPEC = importlib.util.spec_from_file_location(
    "check_pr_release_target", SCRIPT_PATH
)
check = importlib.util.module_from_spec(SPEC)
sys.modules["check_pr_release_target"] = check
SPEC.loader.exec_module(check)


PROJECT_ID = "PVT_kwDOAp2shc4AA8lR"
RELEASE_FIELD_ID = "PVTSSF_lADOAp2shc4AA8lRzgFqH3Y"
RELEASE_MAJOR = 26
RELEASE_04 = f"{RELEASE_MAJOR}.04"
RELEASE_06 = f"{RELEASE_MAJOR}.06"
RELEASE_08 = f"{RELEASE_MAJOR}.08"
RELEASE_10 = f"{RELEASE_MAJOR}.10"
RELEASE_6 = f"{RELEASE_MAJOR}.6"


def issue(number, release):
    return {"name": f"rapidsai/cuml#{number}", "release": release}


def item_with_release(release):
    return {
        "project": {"id": PROJECT_ID},
        "fieldValues": {
            "nodes": [
                {
                    "name": release,
                    "field": {"id": RELEASE_FIELD_ID},
                }
            ]
        },
    }


def pull_request(pr_release, issue_releases, *, has_next_page=False):
    return {
        "projectItems": {"nodes": [item_with_release(pr_release)]},
        "closingIssuesReferences": {
            "pageInfo": {"hasNextPage": has_next_page},
            "nodes": [
                {
                    "number": number,
                    "repository": {"nameWithOwner": "rapidsai/cuml"},
                    "projectItems": {
                        "nodes": (
                            [item_with_release(release)]
                            if release is not None
                            else []
                        )
                    },
                }
                for number, release in issue_releases
            ],
        },
    }


class ReleaseTargetCheckTests(unittest.TestCase):
    def test_release_comparison_is_numeric(self):
        self.assertEqual(check.compare_releases(RELEASE_06, RELEASE_6), 0)
        self.assertLess(check.compare_releases(RELEASE_04, RELEASE_06), 0)
        self.assertGreater(check.compare_releases(RELEASE_10, RELEASE_06), 0)

    def test_pr_release_must_match_target_release(self):
        errors, _ = check.validate(RELEASE_08, RELEASE_06, [], False)

        self.assertIn("does not match the target branch", errors[0])

    def test_landing_before_issue_release_is_allowed(self):
        errors, _ = check.validate(
            RELEASE_04,
            RELEASE_04,
            [issue(456, RELEASE_06)],
            False,
        )

        self.assertEqual(errors, [])

    def test_landing_after_issue_release_is_blocked(self):
        errors, _ = check.validate(
            RELEASE_08,
            RELEASE_08,
            [issue(456, RELEASE_06)],
            False,
        )

        self.assertIn(f"would land in {RELEASE_08}", errors[0])

    def test_multiple_issue_releases_all_must_not_be_late(self):
        errors, _ = check.validate(
            RELEASE_08,
            RELEASE_08,
            [issue(1, RELEASE_06), issue(2, RELEASE_08)],
            False,
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("rapidsai/cuml#1", errors[0])

    def test_missing_issue_release_warns(self):
        errors, warnings = check.validate(
            RELEASE_08,
            RELEASE_08,
            [issue(456, None)],
            False,
        )

        self.assertEqual(errors, [])
        self.assertIn("does not have a Release", warnings[0])

    def test_extracts_pr_and_issue_releases_from_graphql_response(self):
        pr_release, issues, has_more_issues = check.extract(
            pull_request(RELEASE_06, [(456, RELEASE_06), (789, RELEASE_08)]),
            PROJECT_ID,
            RELEASE_FIELD_ID,
        )

        self.assertEqual(pr_release, RELEASE_06)
        self.assertFalse(has_more_issues)
        self.assertEqual(
            [issue["release"] for issue in issues], [RELEASE_06, RELEASE_08]
        )

    def test_more_than_20_closing_issues_fails(self):
        pr_release, issues, has_more_issues = check.extract(
            pull_request(
                RELEASE_06,
                [(number, RELEASE_06) for number in range(1, 21)],
                has_next_page=True,
            ),
            PROJECT_ID,
            RELEASE_FIELD_ID,
        )
        errors, _ = check.validate(
            RELEASE_06,
            pr_release,
            issues,
            has_more_issues,
        )

        self.assertEqual(len(issues), 20)
        self.assertIn("more than 20 closing issue", errors[0])


if __name__ == "__main__":
    unittest.main()
