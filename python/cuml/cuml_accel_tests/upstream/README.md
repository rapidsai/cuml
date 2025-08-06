# Upstream Acceleration Tests

This suite provides infrastructure to run and analyze the test suites of
upstream libraries (scikit-learn, ...) with cuML acceleration support.

## Components

- `run-tests.sh`
  Executes scikit-learn tests using GPU-accelerated paths. Any arguments passed to the script are forwarded directly to pytest.

  Example usage:
  ```bash
  ./scikit-learn/run-tests.sh                     # Run all tests
  ./scikit-learn/run-tests.sh -v -k test_kmeans   # Run specific test with verbosity
  ./scikit-learn/run-tests.sh -x --pdb            # Stop on first failure and debug
  ```

- `summarize-results.py`
  Analyzes test results from an XML report file and prints a summary or generates an xfail list.
  Options:
  - `-v, --verbose`          : Display detailed failure information
  - `-f, --fail-below VALUE` : Set a minimum pass rate threshold (0-100)
  - `--format FORMAT`        : Output format (summary or xfail_list)
  - `--update-xfail-list PATH` : Path to existing xfail list to update
  - `-i, --in-place`        : Update the xfail list file in place
  - `--xpassed ACTION`      : How to handle XPASS tests (keep/remove/mark-flaky)

## Usage

### 1. Run tests and generate report
Run tests and save the report:
```bash
./scikit-learn/run-tests.sh --junitxml=report.xml
```

**Tip**: Run tests in parallel with `-n auto` to use all available CPU cores:
```bash
./scikit-learn/run-tests.sh --junitxml=report.xml -n auto
```

### 2. Analyze results
Generate a summary from the report:
```bash
./summarize-results.py -v -f 80 report.xml
```

## Xfail List

The xfail list (`xfail-list.yaml`) is used to mark tests that are expected to fail. This is useful for:
- Tracking known issues
- Managing test failures during development
- Handling version-specific test failures
- Managing flaky tests that occasionally fail

### Automatic Usage
The `run-tests.sh` script automatically uses an `xfail-list.yaml` file if present in the same directory.

### Generating an Xfail List
The `summarize-results.py` script provides several ways to manage the xfail list:

1. Generate a new xfail list from test results:
```bash
./summarize-results.py --format=xfail_list report.xml > xfail-list.yaml
```

2. Update an existing xfail list (in place):
```bash
./summarize-results.py --update-xfail-list=xfail-list.yaml --in-place report.xml
```

The script handles XPASS tests in three ways (controlled by `--xpassed`):
- `keep`: Preserve all xpassed tests in the list (default)
- `remove`: Remove xpassed tests from the list
- `mark-flaky`: Convert strict xpassed tests to non-strict (flaky)

Example with all options:
```bash
./summarize-results.py --update-xfail-list=xfail-list.yaml --in-place --xpassed=mark-flaky report.xml
```

### Format
The xfail list is a YAML file containing groups of tests to mark as xfail. Each group can include:
- `reason`: Description of why the tests in this group are expected to fail
- `strict`: Whether to enforce xfail (default: true)
- `tests`: List of test IDs in format "module::test_name"
- `condition`: Optional version requirement (e.g., "scikit-learn>=1.5.2")

Example:
```yaml
- reason: "Known issues with sparse inputs"
  strict: true
  tests:
    - "sklearn.linear_model.tests.test_logistic::test_logistic_regression"
    - "sklearn.linear_model.tests.test_ridge::test_ridge_sparse"

- reason: "Unsupported hyperparameters for older scikit-learn version"
  condition: "scikit-learn<1.5.2"
  tests:
    - "sklearn.cluster.tests.test_k_means::test_kmeans_convergence[42-elkan]"
    - "sklearn.cluster.tests.test_k_means::test_kmeans_convergence[42-lloyd]"

- reason: "Flaky tests due to random seed sensitivity"
  strict: false
  tests:
    - "sklearn.ensemble.tests.test_forest::test_random_forest_classifier"
    - "sklearn.ensemble.tests.test_forest::test_random_forest_regressor"
```

**Note on `strict: false`**:
The `strict` flag should be set to `true` by default. Use `strict: false` only for:
- Tests that are genuinely non-deterministic (e.g., due to floating-point arithmetic)
- Tests that fail intermittently due to external factors (e.g., network timeouts)
- Tests that are known to be flaky but cannot be fixed immediately

Ideally, Each use of `strict: false` should include:
- A clear explanation of why the test is non-deterministic
- A plan to fix the underlying issue
- Regular review to ensure the flag is still necessary

### Handling Unmatched Test IDs

The pytest plugin validates that all test IDs in the xfail list correspond to actual tests. When tests don't exist, a `UnmatchedXfailTests` warning is issued.

#### Common Causes
- **Version-specific tests**: Tests that only exist in certain dependency versions
- **Renamed/removed tests**: Tests changed across versions
- **Typographical errors**: Misspelled test IDs

You can suppress the warning during development by removing the pytest argument that
elevates the warning to an error within `run-tests.sh`.

## Recommended workflow for fixing parity issues

### 1. Initial Triage
Start by identifying and categorizing test failures:

```bash
# Get current commit hash
alias gitsha='git rev-parse --short HEAD'

# Run all untriaged tests with --runxfail to actually see all failures
./scikit-learn/run-tests.sh -m cuml_accel_bugs --runxfail --junitxml="report-bugs-$(gitsha).xml" | tee report-bugs-$(gitsha).log
```

### 2. Analyze and Group Failures
Use the summarize-results script to analyze failures:
```bash
# Select the first 10 failures
./summarize-results.py report-bugs-$(gitsha).xml --limit 10 --format=traceback
```

Group similar failures together based on:
- Common error messages
- Related functionality
- Similar root causes

### 3. Update Xfail List with Detailed Markers

Add detailed markers and reasons for each failure group:

```yaml
# Example of a bug that needs fixing
- reason: "Missing scikit-learn interface attributes (components_ and _parameter_constraints)"
  marker: cuml_accel_dbscan_missing_interface
  tests:
    - "sklearn.cluster.tests.test_dbscan::test_dbscan_no_core_samples[csr_array]"
    - "sklearn.cluster.tests.test_dbscan::test_dbscan_no_core_samples[csr_matrix]"
    - "sklearn.tests.test_public_functions::test_class_wrapper_param_validation[sklearn.cluster.dbscan-sklearn.cluster.DBSCAN]"

# Example of a bug in sparse matrix handling
- reason: "Incomplete sparse precomputed distances implementation in NearestNeighbors"
  marker: cuml_accel_dbscan_sparse_precomputed
  tests:
    - "sklearn.cluster.tests.test_dbscan::test_dbscan_sparse_precomputed[False]"
    - "sklearn.cluster.tests.test_dbscan::test_dbscan_sparse_precomputed[True]"
    - "sklearn.cluster.tests.test_dbscan::test_dbscan_sparse_precomputed_different_eps"

# Example of expected divergence (not a bug)
- reason: "GPU-optimized implementation produces different but valid results"
  marker: cuml_accel_dbscan_expected_divergence
  strict: false
  tests:
    - "sklearn.cluster.tests.test_dbscan::test_dbscan_cluster_assignments[float32]"
    - "sklearn.cluster.tests.test_dbscan::test_dbscan_cluster_assignments[float64]"
```

### 4. Debug and Fix Issues
For each failure group:

1. Run tests with the specific marker:
```bash
SELECT=cuml_accel_dbscan_missing_interface
./scikit-learn/run-tests.sh -m "${SELECT}" --runxfail --junitxml="report-${SELECT}-$(gitsha).xml" | tee "test-${SELECT}-$(gitsha).log"
```

2. Investigate the root cause.
3. Implement fixes.

### 5. Verify and Update Xfail List
After fixing issues and committing them, run

1. Run tests again to verify fixes:
```bash
./scikit-learn/run-tests.sh -m "${SELECT}" --junitxml="report-${SELECT}-$(gitsha).xml"
```

2. Update the xfail list:
   - Remove fixed tests from the list
   - Move flaky tests to a separate group with `strict: false`
   - Update reasons for remaining failures

### 6. Document Changes
For each fix:
- Update relevant documentation
- Add comments explaining the fix
- Document any known limitations or trade-offs

### Best Practices
- Keep test groups small and focused
- Use descriptive markers and reasons
- Document root causes and fixes
- Use `strict: false` only for genuinely non-deterministic (flaky) tests
