#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

usage() {
    echo "Usage: $0 [options] REPORT_FILE"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -v, --verbose          Show detailed failure information"
    echo "  -t, --threshold VALUE  Minimum pass rate threshold [0-100] (default: 0)"
    exit 1
}

# Parse command line arguments
THRESHOLD=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        -t|--threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        *)
            if [ -z "${REPORT_FILE:-}" ]; then
                REPORT_FILE="$1"
            else
                echo "Unknown option: $1"
                usage
            fi
            shift
            ;;
    esac
done

if [ -z "${REPORT_FILE:-}" ]; then
    echo "Error: No report file specified"
    usage
fi

if ! [[ "$THRESHOLD" =~ ^[0-9]+(\.[0-9]+)?$ ]] || [ "$(echo "$THRESHOLD > 100" | bc -l)" -eq 1 ]; then
    echo "Error: Threshold must be a number between 0 and 100"
    exit 1
fi

# Extract test statistics using xmllint
total_tests=$(xmllint --xpath "string(/testsuites/testsuite/@tests)" "${REPORT_FILE}")
failures=$(xmllint --xpath "string(/testsuites/testsuite/@failures)" "${REPORT_FILE}")
errors=$(xmllint --xpath "string(/testsuites/testsuite/@errors)" "${REPORT_FILE}")
skipped=$(xmllint --xpath "string(/testsuites/testsuite/@skipped)" "${REPORT_FILE}")
time=$(xmllint --xpath "string(/testsuites/testsuite/@time)" "${REPORT_FILE}")

# Calculate passed tests and pass rate
passed=$((total_tests - failures - errors - skipped))
pass_rate=$(echo "scale=2; 100 * $passed / $total_tests" | bc)

# Print summary
echo "Test Summary:"
echo "  Total Tests: ${total_tests}"
echo "  Passed:      ${passed}"
echo "  Failed:      ${failures}"
echo "  Errors:      ${errors}"
echo "  Skipped:     ${skipped}"
echo "  Pass Rate:   ${pass_rate}%"
echo "  Total Time:  ${time}s"

# List failed tests only in verbose mode
if [ "$((failures + errors))" -gt 0 ] && [ "${VERBOSE}" -eq 1 ]; then
    echo ""
    echo "Failed Tests:"
    xmllint --xpath "//testcase[failure or error]/@name" "${REPORT_FILE}" | tr ' ' '\n' | sed 's/name=//g' | sed 's/"//g' | grep .
fi

# Check if pass rate meets threshold
if [ "$(echo "${pass_rate} < ${THRESHOLD}" | bc -l)" -eq 1 ]; then
    echo ""
    echo "Error: Pass rate ${pass_rate}% is below threshold ${THRESHOLD}%"
    exit 1
fi

# Return failure if there were any failures or errors
exit $((failures + errors))
