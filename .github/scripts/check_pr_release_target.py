#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Any


RELEASE_PATTERN = re.compile(r"^\d+(?:\.\d+)+$")

QUERY = """
query($owner: String!, $repo: String!, $number: Int!) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) {
      projectItems(first: 20) {
        nodes {
          ...ProjectItem
        }
      }
      closingIssuesReferences(first: 20) {
        pageInfo {
          hasNextPage
        }
        nodes {
          number
          repository {
            nameWithOwner
          }
          projectItems(first: 20) {
            nodes {
              ...ProjectItem
            }
          }
        }
      }
    }
  }
}

fragment ProjectItem on ProjectV2Item {
  project {
    id
  }
  fieldValues(first: 100) {
    nodes {
      ... on ProjectV2ItemFieldSingleSelectValue {
        name
        field {
          ... on ProjectV2SingleSelectField {
            id
          }
        }
      }
    }
  }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr-number", required=True, type=int)
    parser.add_argument("--target-release", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--release-field-id", required=True)
    return parser.parse_args()


def release_key(release: str) -> tuple[int, ...] | None:
    if not RELEASE_PATTERN.fullmatch(release.strip()):
        return None
    return tuple(int(part) for part in release.split("."))


def compare_releases(left: str, right: str) -> int:
    left_key = release_key(left)
    right_key = release_key(right)
    if left_key is None:
        raise ValueError(f"Unrecognized release value: {left!r}")
    if right_key is None:
        raise ValueError(f"Unrecognized release value: {right!r}")

    width = max(len(left_key), len(right_key))
    left_key += (0,) * (width - len(left_key))
    right_key += (0,) * (width - len(right_key))
    return (left_key > right_key) - (left_key < right_key)


def fetch_pr(repo: str, number: int) -> dict[str, Any]:
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise RuntimeError("GH_TOKEN is not set")

    owner, name = repo.split("/", 1)
    payload = {
        "query": QUERY,
        "variables": {"owner": owner, "repo": name, "number": number},
    }
    request = urllib.request.Request(
        "https://api.github.com/graphql",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"GitHub GraphQL request failed: {exc.code} {body}"
        ) from exc

    if data.get("errors"):
        raise RuntimeError(
            f"GitHub GraphQL returned errors: {json.dumps(data['errors'])}"
        )

    pr = data.get("data", {}).get("repository", {}).get("pullRequest")
    if not pr:
        raise RuntimeError("GraphQL response did not include the pull request")
    return pr


def release_value(
    node: dict[str, Any], project_id: str, release_field_id: str
) -> str | None:
    for item in node.get("projectItems", {}).get("nodes", []):
        if item.get("project", {}).get("id") != project_id:
            continue
        for value in item.get("fieldValues", {}).get("nodes", []):
            if value.get("field", {}).get("id") == release_field_id:
                return value.get("name")
    return None


def extract(
    pr: dict[str, Any], project_id: str, release_field_id: str
) -> tuple[str | None, list[dict[str, str | None]], bool]:
    closing_issues = pr.get("closingIssuesReferences", {})
    issues = []
    for issue in closing_issues.get("nodes", []):
        issues.append(
            {
                "name": (
                    f"{issue.get('repository', {}).get('nameWithOwner')}"
                    f"#{issue['number']}"
                ),
                "release": release_value(issue, project_id, release_field_id),
            }
        )
    return (
        release_value(pr, project_id, release_field_id),
        issues,
        bool(closing_issues.get("pageInfo", {}).get("hasNextPage")),
    )


def validate(
    target: str,
    pr_release: str | None,
    issues: list[dict[str, str | None]],
    has_more_issues: bool,
) -> tuple[list[str], list[str]]:
    errors = []
    warnings = []

    if not target:
        errors.append("Target release is not set.")
    elif release_key(target) is None:
        errors.append(f"Target release {target!r} is not a numeric release.")

    if pr_release is None:
        errors.append(
            "The PR project item does not have a Release field value."
        )
    elif release_key(pr_release) is None:
        errors.append(f"PR Release {pr_release!r} is not a numeric release.")

    if target and pr_release is not None:
        try:
            if compare_releases(pr_release, target) != 0:
                errors.append(
                    "PR Release does not match the target branch: "
                    f"PR Release is {pr_release}, but target release is "
                    f"{target}."
                )
        except ValueError:
            pass

    if not issues:
        warnings.append(
            "No closing issue references found; issue release check skipped."
        )
    if has_more_issues:
        errors.append(
            "The PR has more than 20 closing issue references; release "
            "validation only checks the first 20."
        )

    if pr_release is None or release_key(pr_release) is None:
        return errors, warnings

    for issue in issues:
        issue_release = issue["release"]
        if issue_release is None:
            warnings.append(
                f"{issue['name']} does not have a Release field value."
            )
        elif release_key(issue_release) is None:
            warnings.append(
                f"{issue['name']} Release {issue_release!r} is not a numeric release."
            )
        elif compare_releases(pr_release, issue_release) > 0:
            errors.append(
                f"{issue['name']} is scheduled for {issue_release}, but the PR "
                f"would land in {pr_release}."
            )

    return errors, warnings


def main() -> int:
    args = parse_args()
    pr_release, issues, has_more_issues = extract(
        fetch_pr(args.repo, args.pr_number),
        args.project_id,
        args.release_field_id,
    )
    errors, warnings = validate(
        args.target_release,
        pr_release,
        issues,
        has_more_issues,
    )

    print(f"Target release: {args.target_release or 'unset'}")
    print(f"PR Release: {pr_release or 'unset'}")
    if issues:
        print("Linked issues:")
        for issue in issues:
            print(f"- {issue['name']}: {issue['release'] or 'unset'}")
    else:
        print("Linked issues: none")

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
