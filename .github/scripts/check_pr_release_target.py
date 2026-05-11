#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate that a PR's project Release matches its target branch.

The PR project item declares the release a PR will land in. Linked issues may
target the same or a later release, but never an earlier one.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROJECT_ID = "PVT_kwDOAp2shc4AA8lR"
DEFAULT_RELEASE_FIELD_ID = "PVTSSF_lADOAp2shc4AA8lRzgFqH3Y"
DEFAULT_RELEASE_FIELD_NAME = "Release"

RELEASE_PATTERN = re.compile(r"^\d+(?:\.\d+)+$")

PR_RELEASE_QUERY = """
query($owner: String!, $repo: String!, $number: Int!) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) {
      number
      title
      url
      baseRefName
      projectItems(first: 20) {
        nodes {
          ...ProjectItemFields
        }
      }
      closingIssuesReferences(first: 20) {
        nodes {
          id
          number
          title
          url
          state
          repository {
            nameWithOwner
          }
          projectItems(first: 20) {
            nodes {
              ...ProjectItemFields
            }
          }
        }
      }
    }
  }
}

fragment ProjectItemFields on ProjectV2Item {
  id
  project {
    id
    title
    number
  }
  fieldValues(first: 100) {
    nodes {
      __typename
      ... on ProjectV2ItemFieldSingleSelectValue {
        name
        field {
          ... on ProjectV2SingleSelectField {
            id
            name
          }
        }
      }
    }
  }
}
"""


@dataclass(frozen=True)
class LinkedIssue:
    number: int
    title: str
    url: str
    repository: str
    release: str | None


@dataclass(frozen=True)
class ValidationResult:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    summary: str

    @property
    def ok(self) -> bool:
        return not self.errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that PR and linked issue Release fields are consistent."
    )
    parser.add_argument(
        "--repo", required=True, help="Repository as owner/name."
    )
    parser.add_argument("--pr-number", required=True, type=int)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument(
        "--version-file",
        default="VERSION",
        help="VERSION file from the PR base branch checkout.",
    )
    parser.add_argument(
        "--target-release", help="Override derived target release."
    )
    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID)
    parser.add_argument("--release-field-id", default=DEFAULT_RELEASE_FIELD_ID)
    parser.add_argument(
        "--release-field-name", default=DEFAULT_RELEASE_FIELD_NAME
    )
    parser.add_argument(
        "--json",
        help="Read GraphQL response JSON from a file instead of calling GitHub.",
    )
    parser.add_argument(
        "--token-env",
        default="GH_TOKEN",
        help="Environment variable containing a GitHub token.",
    )
    parser.add_argument(
        "--require-linked-issue",
        action="store_true",
        help="Fail if the PR does not have closing issue references.",
    )
    parser.add_argument(
        "--require-issue-release",
        action="store_true",
        help="Fail if a linked issue is missing the Release field.",
    )
    return parser.parse_args()


def derive_target_release(base_ref: str, version_file: str) -> str | None:
    if base_ref.startswith("release/"):
        return base_ref.removeprefix("release/")
    if base_ref == "main":
        version_path = Path(version_file)
        if not version_path.is_file():
            return None
        version = (
            version_path.read_text(encoding="utf-8").splitlines()[0].strip()
        )
        return ".".join(version.split(".")[:2])
    return None


def release_key(release: str) -> tuple[int, ...] | None:
    release = release.strip()
    if not RELEASE_PATTERN.fullmatch(release):
        return None
    return tuple(int(part) for part in release.split("."))


def compare_releases(left: str, right: str) -> int:
    left_key = release_key(left)
    right_key = release_key(right)
    if left_key is None:
        raise ValueError(f"Unrecognized release value: {left!r}")
    if right_key is None:
        raise ValueError(f"Unrecognized release value: {right!r}")
    max_len = max(len(left_key), len(right_key))
    padded_left = left_key + (0,) * (max_len - len(left_key))
    padded_right = right_key + (0,) * (max_len - len(right_key))
    return (padded_left > padded_right) - (padded_left < padded_right)


def fetch_pr_data(repo: str, pr_number: int, token_env: str) -> dict[str, Any]:
    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(f"{token_env} is not set")

    owner, name = repo.split("/", 1)
    payload = {
        "query": PR_RELEASE_QUERY,
        "variables": {"owner": owner, "repo": name, "number": pr_number},
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
    return data


def select_project_item(
    node: dict[str, Any], project_id: str
) -> dict[str, Any] | None:
    for item in node.get("projectItems", {}).get("nodes", []):
        if item and item.get("project", {}).get("id") == project_id:
            return item
    return None


def release_field_value(
    item: dict[str, Any] | None,
    release_field_id: str,
    release_field_name: str,
) -> str | None:
    if not item:
        return None
    for field_value in item.get("fieldValues", {}).get("nodes", []):
        field = field_value.get("field") or {}
        if (
            field.get("id") == release_field_id
            or field.get("name") == release_field_name
        ):
            return field_value.get("name")
    return None


def extract_pr_and_issues(
    data: dict[str, Any],
    project_id: str,
    release_field_id: str,
    release_field_name: str,
) -> tuple[dict[str, Any], str | None, list[LinkedIssue]]:
    pr = data.get("data", {}).get("repository", {}).get("pullRequest")
    if not pr:
        raise RuntimeError("GraphQL response did not include the pull request")

    pr_release = release_field_value(
        select_project_item(pr, project_id),
        release_field_id,
        release_field_name,
    )
    issues = []
    for issue in pr.get("closingIssuesReferences", {}).get("nodes", []):
        issue_release = release_field_value(
            select_project_item(issue, project_id),
            release_field_id,
            release_field_name,
        )
        issues.append(
            LinkedIssue(
                number=issue["number"],
                title=issue.get("title") or "",
                url=issue.get("url") or "",
                repository=issue.get("repository", {}).get("nameWithOwner")
                or "",
                release=issue_release,
            )
        )
    return pr, pr_release, issues


def validate_release_target(
    *,
    target_release: str | None,
    pr_release: str | None,
    issues: list[LinkedIssue],
    base_ref: str,
    require_linked_issue: bool = False,
    require_issue_release: bool = False,
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if not target_release:
        errors.append(
            f"Could not derive a target release from base branch {base_ref!r}."
        )
    elif release_key(target_release) is None:
        errors.append(
            f"Target release {target_release!r} is not a numeric release."
        )

    if not pr_release:
        errors.append(
            "The PR project item does not have a Release field value."
        )
    elif release_key(pr_release) is None:
        errors.append(f"PR Release {pr_release!r} is not a numeric release.")

    if target_release and pr_release:
        try:
            if compare_releases(pr_release, target_release) != 0:
                errors.append(
                    "PR Release does not match the target branch: "
                    f"PR Release is {pr_release}, but base branch {base_ref} "
                    f"lands in {target_release}."
                )
        except ValueError:
            pass

    if require_linked_issue and not issues:
        errors.append("The PR does not have any closing issue references.")
    elif not issues:
        warnings.append(
            "No closing issue references found; issue release check skipped."
        )

    for issue in issues:
        issue_name = (
            f"{issue.repository}#{issue.number}"
            if issue.repository
            else f"#{issue.number}"
        )
        if not issue.release:
            message = f"{issue_name} does not have a Release field value."
            if require_issue_release:
                errors.append(message)
            else:
                warnings.append(message)
            continue
        if release_key(issue.release) is None:
            message = f"{issue_name} Release {issue.release!r} is not a numeric release."
            if require_issue_release:
                errors.append(message)
            else:
                warnings.append(message)
            continue
        if pr_release and release_key(pr_release) is not None:
            if compare_releases(pr_release, issue.release) > 0:
                errors.append(
                    f"{issue_name} is scheduled for {issue.release}, but the PR "
                    f"would land in {pr_release}."
                )

    summary_lines = [
        f"Target release: {target_release or 'unknown'}",
        f"PR Release: {pr_release or 'unset'}",
    ]
    if issues:
        summary_lines.append("Linked issues:")
        for issue in issues:
            issue_name = (
                f"{issue.repository}#{issue.number}"
                if issue.repository
                else f"#{issue.number}"
            )
            summary_lines.append(f"- {issue_name}: {issue.release or 'unset'}")
    else:
        summary_lines.append("Linked issues: none")

    return ValidationResult(
        tuple(errors), tuple(warnings), "\n".join(summary_lines)
    )


def write_step_summary(result: ValidationResult) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    lines = [
        "## PR Release Target Check",
        "",
        "```text",
        result.summary,
        "```",
        "",
    ]
    if result.errors:
        lines.extend(["### Errors", ""])
        lines.extend(f"- {error}" for error in result.errors)
        lines.append("")
    if result.warnings:
        lines.extend(["### Warnings", ""])
        lines.extend(f"- {warning}" for warning in result.warnings)
        lines.append("")
    Path(summary_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    target_release = args.target_release or derive_target_release(
        args.base_ref,
        args.version_file,
    )

    if args.json:
        data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    else:
        data = fetch_pr_data(args.repo, args.pr_number, args.token_env)

    _, pr_release, issues = extract_pr_and_issues(
        data,
        args.project_id,
        args.release_field_id,
        args.release_field_name,
    )
    result = validate_release_target(
        target_release=target_release,
        pr_release=pr_release,
        issues=issues,
        base_ref=args.base_ref,
        require_linked_issue=args.require_linked_issue,
        require_issue_release=args.require_issue_release,
    )

    print(result.summary)
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for error in result.errors:
        print(f"error: {error}", file=sys.stderr)
    write_step_summary(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
