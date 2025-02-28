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
  - `-p, --path`             : Specify the scikit-learn source location
  - `--relevant-only`        : Run only tests that are relevant for GPU acceleration
  - `-- [pytest-args]`       : Pass additional arguments directly to pytest

- `summarize-results.sh`
  Analyzes test results from an XML report file and prints a summary.
  Options:
  - `-v, --verbose`          : Display detailed failure information
  - `-f, --fail-below VALUE`  : Set a minimum pass rate threshold (0-100)

## Usage

### 1. Build scikit-learn
```bash
./build.sh --scikit-learn-version 1.5.2
```

### 2. Run tests
For GPU-accelerated tests only:
```bash
./run-tests.sh --relevant-only -p ./scikit-learn -- [additional pytest options]
```
Or, to run all tests:
```bash
./run-tests.sh -p ./scikit-learn -- [additional pytest options]
```

### 3. Summarize test results
Generate a summary from the XML report with a pass rate threshold:
```bash
./summarize-results.sh -v -f 80 report.xml
```

## CI Integration

This suite integrates with the cuML CI pipeline via scripts such as `ci/test_python_scikit_learn_tests.sh`.
