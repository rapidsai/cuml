# scikit-learn Acceleration Tests

This suite provides infrastructure to run and analyze tests for scikit-learn with cuML acceleration support.

## Components

- `run-tests.sh`
  Executes scikit-learn tests using GPU-accelerated paths. Any arguments passed to the script are forwarded directly to pytest.

  Example usage:
  ```bash
  ./run-tests.sh                     # Run all tests
  ./run-tests.sh -v -k test_kmeans   # Run specific test with verbosity
  ./run-tests.sh -x --pdb            # Stop on first failure and debug
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
./run-tests.sh --junitxml=report.xml
```

**Tip**: Run tests in parallel with `-n auto` to use all available CPU cores:
```bash
./run-tests.sh --junitxml=report.xml -n auto
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
