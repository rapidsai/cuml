# scikit-learn Acceleration Tests

This suite provides infrastructure to build, run, and analyze tests for scikit-learn with cuML acceleration support.

## Components

- `build.sh`
  Clones and builds the scikit-learn repository.
  Options:
  - `--scikit-learn-version` : Specify the scikit-learn version to test (default: 1.5.2)
  - `-p, --path`             : Custom path for the scikit-learn source (default: ./scikit-learn)

- `run-tests.sh`
  Executes scikit-learn tests using GPU-accelerated paths.
  Options:
  - `-p, --path`             : Specify the scikit-learn source location (default: ./scikit-learn)
  - `--select [minimal|relevant]` : Select tests based on predefined groups from a YAML config (default: run all tests)
  - `-- [pytest-args]`       : Pass additional arguments directly to pytest

- `summarize-results.sh`
  Analyzes test results from an XML report file and prints a summary.
  Options:
  - `-v, --verbose`          : Display detailed failure information
  - `-f, --fail-below VALUE`  : Set a minimum pass rate threshold (0-100)

## Usage

### Important Note
The `run-tests.sh` script must be executed from within its parent directory. This ensures proper resolution of test paths and configuration files.

### 1. Build scikit-learn
```bash
./build.sh --scikit-learn-version 1.5.2
```

### 2. Run tests
For a specific test selection:
```bash
./run-tests.sh --select minimal -p ./scikit-learn
```
Or, to run all tests:
```bash
./run-tests.sh -p ./scikit-learn
```

You can also pass additional pytest arguments:
```bash
./run-tests.sh --select relevant -p ./scikit-learn -- -v -k "test_logistic"
```

### 3. Summarize test results
Generate a summary from the XML report with a pass rate threshold:
```bash
./summarize-results.sh -v -f 80 report.xml
```

## CI Integration

This suite integrates with the cuML CI pipeline via scripts such as `ci/test_python_scikit_learn_tests.sh`.
