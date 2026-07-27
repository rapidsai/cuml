# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path

SUMMARIZER = Path(__file__).with_name("summarize-results.py")

REPORT = """\
<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="4" errors="0" failures="0" skipped="2">
    <testcase classname="examples" name="passes_1"/>
    <testcase classname="examples" name="passes_2"/>
    <testcase classname="examples" name="network_xfail">
      <skipped type="pytest.xfail"
               message="reason: Network error: urllib.error.HTTPError"/>
    </testcase>
    <testcase classname="examples" name="timeout_xfail">
      <skipped type="pytest.xfail"
               message="reason: Timeout: example exceeded 300s"/>
    </testcase>
  </testsuite>
</testsuites>
"""


def run_summarizer(report, *args):
    return subprocess.run(
        [sys.executable, SUMMARIZER, *args, report],
        capture_output=True,
        text=True,
    )


def parse_summary(output):
    return dict(
        line.strip().split(":", maxsplit=1)
        for line in output.splitlines()
        if ":" in line
    )


def test_all_xfails_count_against_pass_rate_by_default(tmp_path):
    report = tmp_path / "report.xml"
    report.write_text(REPORT)

    result = run_summarizer(report)
    summary = parse_summary(result.stdout)

    assert result.returncode == 0
    assert summary["Excluded XFailed"].strip() == "0"
    assert summary["Pass Rate"].strip() == "50.00%"
    assert summary["Total Pass Rate"].strip() == "50.00%"


def test_matching_xfail_is_excluded_from_pass_rate_denominator(tmp_path):
    report = tmp_path / "report.xml"
    report.write_text(REPORT)

    result = run_summarizer(
        report,
        "--fail-below",
        "60",
        "--exclude-xfail-reason",
        "Network error:",
    )
    summary = parse_summary(result.stdout)

    assert result.returncode == 0
    assert summary["XFailed"].strip() == "2"
    assert summary["Excluded XFailed"].strip() == "1"
    assert summary["Pass Rate"].strip() == "66.67%"
    assert summary["Total Pass Rate"].strip() == "50.00%"


def test_timeout_xfail_remains_in_pass_rate_denominator(tmp_path):
    report = tmp_path / "report.xml"
    report.write_text(REPORT)

    result = run_summarizer(
        report,
        "--fail-below",
        "70",
        "--exclude-xfail-reason",
        "Network error:",
    )
    summary = parse_summary(result.stdout)

    assert result.returncode == 1
    assert summary["Excluded XFailed"].strip() == "1"
    assert summary["Pass Rate"].strip() == "66.67%"
    assert "below threshold 70.0%" in result.stdout


def test_total_pass_rate_has_separate_threshold(tmp_path):
    report = tmp_path / "report.xml"
    report.write_text(REPORT)

    result = run_summarizer(
        report,
        "--fail-below",
        "60",
        "--total-fail-below",
        "55",
        "--exclude-xfail-reason",
        "Network error:",
    )
    summary = parse_summary(result.stdout)

    assert result.returncode == 1
    assert summary["Pass Rate"].strip() == "66.67%"
    assert summary["Total Pass Rate"].strip() == "50.00%"
    assert "Total pass rate 50.00% is below threshold 55.0%" in result.stdout
