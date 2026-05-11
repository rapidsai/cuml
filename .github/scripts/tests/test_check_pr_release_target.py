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


def item_with_release(release):
    return {
        "project": {"id": PROJECT_ID},
        "fieldValues": {
            "nodes": [
                {
                    "__typename": "ProjectV2ItemFieldSingleSelectValue",
                    "name": release,
                    "field": {"id": RELEASE_FIELD_ID, "name": "Release"},
                }
            ]
        },
    }


def graphql_response(pr_release, issue_releases):
    return {
        "data": {
            "repository": {
                "pullRequest": {
                    "number": 123,
                    "title": "Example PR",
                    "url": "https://github.com/rapidsai/cuml/pull/123",
                    "baseRefName": "main",
                    "projectItems": {"nodes": [item_with_release(pr_release)]},
                    "closingIssuesReferences": {
                        "nodes": [
                            {
                                "number": number,
                                "title": f"Issue {number}",
                                "url": f"https://github.com/rapidsai/cuml/issues/{number}",
                                "repository": {
                                    "nameWithOwner": "rapidsai/cuml"
                                },
                                "projectItems": {
                                    "nodes": (
                                        [item_with_release(release)]
                                        if release is not None
                                        else []
                                    )
                                },
                            }
                            for number, release in issue_releases
                        ]
                    },
                }
            }
        }
    }


class ReleaseTargetCheckTests(unittest.TestCase):
    def test_release_comparison_is_numeric(self):
        self.assertEqual(check.compare_releases(RELEASE_06, RELEASE_6), 0)
        self.assertLess(check.compare_releases(RELEASE_04, RELEASE_06), 0)
        self.assertGreater(check.compare_releases(RELEASE_10, RELEASE_06), 0)

    def test_pr_release_must_match_target_release(self):
        result = check.validate_release_target(
            target_release=RELEASE_08,
            pr_release=RELEASE_06,
            issues=[],
            base_ref="main",
        )

        self.assertFalse(result.ok)
        self.assertIn("does not match the target branch", result.errors[0])

    def test_landing_before_issue_release_is_allowed(self):
        result = check.validate_release_target(
            target_release=RELEASE_04,
            pr_release=RELEASE_04,
            issues=[
                check.LinkedIssue(
                    number=456,
                    title="Future work",
                    url="",
                    repository="rapidsai/cuml",
                    release=RELEASE_06,
                )
            ],
            base_ref=f"release/{RELEASE_04}",
        )

        self.assertTrue(result.ok, result.errors)

    def test_landing_after_issue_release_is_blocked(self):
        result = check.validate_release_target(
            target_release=RELEASE_08,
            pr_release=RELEASE_08,
            issues=[
                check.LinkedIssue(
                    number=456,
                    title="Current work",
                    url="",
                    repository="rapidsai/cuml",
                    release=RELEASE_06,
                )
            ],
            base_ref="main",
        )

        self.assertFalse(result.ok)
        self.assertIn(f"would land in {RELEASE_08}", result.errors[0])

    def test_multiple_issue_releases_all_must_not_be_late(self):
        result = check.validate_release_target(
            target_release=RELEASE_08,
            pr_release=RELEASE_08,
            issues=[
                check.LinkedIssue(1, "Early", "", "rapidsai/cuml", RELEASE_06),
                check.LinkedIssue(2, "Same", "", "rapidsai/cuml", RELEASE_08),
            ],
            base_ref="main",
        )

        self.assertFalse(result.ok)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("rapidsai/cuml#1", result.errors[0])

    def test_missing_issue_release_warns_by_default(self):
        result = check.validate_release_target(
            target_release=RELEASE_08,
            pr_release=RELEASE_08,
            issues=[
                check.LinkedIssue(456, "No release", "", "rapidsai/cuml", None)
            ],
            base_ref="main",
        )

        self.assertTrue(result.ok, result.errors)
        self.assertIn("does not have a Release", result.warnings[0])

    def test_missing_issue_release_can_be_required(self):
        result = check.validate_release_target(
            target_release=RELEASE_08,
            pr_release=RELEASE_08,
            issues=[
                check.LinkedIssue(456, "No release", "", "rapidsai/cuml", None)
            ],
            base_ref="main",
            require_issue_release=True,
        )

        self.assertFalse(result.ok)
        self.assertIn("does not have a Release", result.errors[0])

    def test_extracts_pr_and_issue_releases_from_graphql_response(self):
        data = graphql_response(
            RELEASE_06, [(456, RELEASE_06), (789, RELEASE_08)]
        )

        _, pr_release, issues = check.extract_pr_and_issues(
            data,
            PROJECT_ID,
            RELEASE_FIELD_ID,
            "Release",
        )

        self.assertEqual(pr_release, RELEASE_06)
        self.assertEqual(
            [issue.release for issue in issues], [RELEASE_06, RELEASE_08]
        )


if __name__ == "__main__":
    unittest.main()
