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

- `summarize-results.sh`
  Analyzes test results from an XML report file and prints a summary.
  Options:
  - `-v, --verbose`          : Display detailed failure information
  - `-f, --fail-below VALUE`  : Set a minimum pass rate threshold (0-100)

## Usage

### 1. Run tests
Run all tests:
```bash
./run-tests.sh
```

Run specific tests using pytest arguments:
```bash
./run-tests.sh -v -k "test_logistic"
```

### 2. Summarize test results
Generate a summary from the XML report with a pass rate threshold:
```bash
./summarize-results.sh -v -f 80 report.xml
```
