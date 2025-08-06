# cuML Python Developer Guide

This document provides comprehensive guidelines and best practices for contributing to the cuML Python library, the machine learning library within the CUDA and RAPIDS ecosystem. As an evolving document, we welcome contributions, clarifications, and issue reports to help maintain and improve these guidelines.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Getting Started](#getting-started)
3. [Coding Style](#coding-style)
4. [Documentation](#documentation)
5. [Testing and Unit Testing](#testing-and-unit-testing)
6. [Memory Management](#memory-management)
7. [Thread Safety](#thread-safety)
8. [Creating New Estimators](#creating-new-estimators)
9. [Deprecation Policy](#deprecation-policy)
10. [Logging](#logging)
11. [Multi-GPU Support](#multi-gpu-support)
12. [Benchmarking](#benchmarking)

## Prerequisites

Before diving into Python development for cuML, please ensure you have:

1. Reviewed our [contribution guidelines](../../CONTRIBUTING.md) for general project standards
2. Read the [Python cuML README](../../python/README.md) for setup and installation instructions

If you are working on C++/CUDA code or need to understand the underlying implementation details, you should also familiarize yourself with the [C++ Developer Guide](../cpp/DEVELOPER_GUIDE.md).

## Getting Started

The cuML Python library provides a scikit-learn style API for GPU-accelerated machine learning algorithms. This guide focuses on Python-specific development practices, while maintaining consistency with the underlying C++/CUDA implementations.

## Coding Style

The majority of style guidelines are enforced through pre-commit hooks. Python code must be formatted following the PEP8 style as defined by black and isort.

See [Documentation](#documentation) for guidelines on doc-string formatting.

## Documentation
Doc-string documentation should follow the [NumPy docstring style guide](https://numpydoc.readthedocs.io/en/stable/format.html) for documenting interfaces. This provides a consistent format that is both human-readable and machine-parseable.

### Docstring Sections
The docstring should include the following sections in order:

1. **Short Summary**
   - A one-line summary that does not use variable names or the function name
   - Example: `"""The sum of two numbers."""`

2. **Extended Summary**
   - A few sentences giving an extended description
   - Should clarify functionality, not implementation details

3. **Parameters**
   - Description of function arguments, keywords and their types
   - Format:
   ```python
   Parameters
   ----------
   x : type
       Description of parameter `x`.
   y : type, optional
       Description of parameter `y` (with type not specified).
   ```

4. **Returns**
   - Description of returned values and their types
   - Format:
   ```python
   Returns
   -------
   int
       Description of anonymous integer return value.
   ```

5. **Other Sections** (as needed)
   - Raises
   - See Also
   - Notes
   - References
   - Examples

### Examples
Examples must be written in the form of an interactive REPL. The examples in the
documentation are checked through doctest. To skip the check for an example's
output, use the command `# doctest: +SKIP`. Examples subject to numerical
imprecision, or that can't be reproduced consistently should be skipped.

Example docstring format:
```python
def function_name(param1, param2):
    """Short summary.

    Extended summary if needed.

    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type, optional
        Description of param2

    Returns
    -------
    type
        Description of return value

    Examples
    --------
    >>> function_name(1, 2)
    result
    """
    pass
```

### Best Practices
1. Use reStructuredText (reST) syntax for formatting
2. Keep docstring lines to 75 characters for readability in text terminals
3. Enclose parameter names in single backticks when referring to them
4. Use double backticks for inline code
5. Include type information for all parameters and return values
6. Document default values for optional parameters
7. Use the `@generate_docstring` decorator for common parameter documentation
8. Include examples that demonstrate typical usage

For more details, refer to the [NumPy docstring style guide](https://numpydoc.readthedocs.io/en/stable/format.html).

## Testing and Unit Testing
We use [pytest](https://docs.pytest.org/en/latest/) for writing and running tests. To see existing examples, refer to any of the `test_*.py` files in the folder `cuml/tests`.

### Test Organization
- Keep all tests for a single estimator in one file, with exceptions for:
  - Performance testing/benchmarking
  - Generic estimator checks (e.g., `test_base.py`)
- Use small, focused datasets for correctness testing
- Only explicitly parametrize dataset size when it triggers alternate code paths

### Test Input Generation
We support three main approaches for test input generation:

1. **Fixtures** (`@pytest.fixture`):
   - For shared setup/teardown code and resources
   - Examples: random seeds, clients, loading test datasets
   ```python
   @pytest.fixture(scope="module")
   def random_state():
       return 42
   ```

2. **Parametrization** (`@pytest.mark.parametrize`):
   - For testing specific input combinations
   - Good for hyperparameters and configurations that we need to test _exhaustively_
   ```python
   @pytest.mark.parametrize("solver", ["svd", "eig"])
   def test_estimator(solver):
       pass
   ```

3. **Hypothesis** (`@given`):
   - For property-based testing with random inputs
   - Must include at least one `@example` for deterministic testing
   - Preferred for dataset generation and most hyperparameter testing
   ```python
   @example(dataset=small_regression_dataset(np.float32), alpha=floats(0.1, 10.0))
   @given(dataset=standard_regression_datasets(), alpha=1.0)
   def test_estimator(dataset, alpha):
       pass
   ```

### Test Parameter Levels

You can mark test parameters for different scales with (`unit_param`, `quality_param`, and `stress_param`).

_Note: For dataset scaling, prefer using hypothesis, e.g. with `standard_regression_datasets()`._

We provide three test parameter levels:

1. **Unit Tests** (`unit_param`): Small values for quick, basic functionality testing
   ```python
   unit_param(2)  # For number of components
   ```

2. **Quality Tests** (`quality_param`): Medium values for thorough testing
   ```python
   quality_param(10)  # For number of components
   ```

3. **Stress Tests** (`stress_param`): Large values for performance testing
   ```python
   stress_param(100)  # For number of components
   ```

Control via these pytest options:
- `--run_unit`: Unit tests (default)
- `--run_quality`: Quality tests
- `--run_stress`: Stress tests

### Testing Guidelines

1. **Accuracy Testing**
   - Compare against recorded reference values when possible
   - Document origin of reference values
   - Use appropriate quality metrics for equivalent but different results
   - Ensure reproducibility rather than using retry logic

2. **Minimize resources**
   - Use minimal dataset sizes
   - Only test different scales if they would actually hit different code paths

3. **Best Practices**
   - Write small, focused tests
   - Avoid duplication between test files
   - Choose appropriate input generation method
   - Make tests reproducible
   - Document test assumptions and requirements

### Running Tests
Tests must be run from the `python/cuml/` directory or one of its subdirectories. First build the package, then execute tests.

```bash
./build.sh
cd python/cuml/
pytest  # Run all tests
```

Common options:
- `pytest cuml/tests/test_kmeans.py` - Run specific file
- `pytest -k "test_kmeans"` - Run tests matching pattern
- `pytest --run_unit` - Run only unit tests
- `pytest -v` - Verbose output

Running pytest from outside the `python/cuml/` directory will result in import errors.

## Memory Management

cuML uses RMM (RAPIDS Memory Manager) for GPU memory management and provides a flexible memory management system through the `CumlArray` class. Here are the key points:

1. **Memory Allocation Best Practices**
- Do not use RMM directly to allocate memory
- Use the `CumlArray` class for array-like data allocation
- Use utility functions from `internals.memory_utils` for CuPy array instantiation
- Let cuML handle memory management through its internal mechanisms

2. **Memory Types**
cuML supports several memory types through the `MemoryType` enum:
- `device`: GPU memory for CUDA operations
- `host`: CPU memory for host operations
- `managed`: Unified memory accessible from both CPU and GPU
- `mirror`: Memory type that mirrors the input data's memory type

3. **CumlArray Usage**
The `CumlArray` class provides a unified interface for array data:
```python
from cuml.internals.array import CumlArray

# Create arrays with specific memory types
arr = CumlArray.empty(shape=(1000,), dtype='float32', mem_type='device')
arr = CumlArray.zeros(shape=(1000,), dtype='float32', mem_type='host')

# Convert between memory types
device_arr = host_arr.to_mem_type('device')
host_arr = device_arr.to_mem_type('host')
```

4. **Memory Type Conversion**
- Use `to_output()` for format conversion:
  ```python
  # Convert to different formats
  cupy_arr = arr.to_output('cupy')
  numpy_arr = arr.to_output('numpy')
  cudf_series = arr.to_output('series')
  ```
- Use `to_mem_type()` for explicit memory type conversion
- Consider memory overhead when converting between types
- Supported output types:
  - 'array': CuPy/NumPy arrays
  - 'numba': Numba device arrays
  - 'dataframe': cuDF/Pandas DataFrames
  - 'series': cuDF/Pandas Series

5. **Context Management**
Use context managers for temporary memory type changes:
```python
from cuml.internals.memory_utils import using_memory_type

with using_memory_type('device'):
    # Operations using device memory
    pass
```

6. **Memory Type Detection**
Detect memory type of input data:
```python
from cuml.internals.memory_utils import determine_array_memtype

mem_type = determine_array_memtype(array)
```

7. **Additional Considerations**
- Minimize memory transfers between host and device
- Use appropriate memory types for your operations
- Consider memory layout (C/F order) for optimal performance
- Handle memory allocation failures gracefully

## Thread Safety

Algorithms implemented in C++/CUDA should be implemented in a thread-safe manner. The Python code is generally not thread safe.
Refer to the section on thread safety in [C++ DEVELOPER_GUIDE.md](../cpp/DEVELOPER_GUIDE.md#thread-safety)

## Creating New Estimators

When implementing a new estimator in cuML, follow these key steps:

1. Ensure the algorithm is implemented in C++/CUDA first (see [C++ Developer Guide](../cpp/DEVELOPER_GUIDE.md))
2. Create a Python wrapper class that:
   - Follows scikit-learn's API design patterns
   - Inherits from `cuml.Base`
   - Is placed in the appropriate subdirectory matching scikit-learn's structure

For detailed implementation guidelines, including file organization, API design, and best practices, refer to the [Estimator Guide](ESTIMATOR_GUIDE.md).

## Deprecation Policy

cuML follows the policy of deprecating code for one release prior to removal. This applies
to publicly accessible functions, classes, methods, attributes and parameters. During the
deprecation cycle the old name or value is still supported, but will raise a deprecation
warning when it is used.

Code in cuML should not use deprecated cuML APIs.

```python
warnings.warn(
    (
        "Attribute `foo` was deprecated in version 25.06 and will be"
        " removed in 25.08. Use `metric` instead."
    ),
    FutureWarning,
)
```

The warning message should always give both the version in which the deprecation was introduced
and the version in which the old behavior will be removed. The message should also include
a brief explanation of the change and point users to an alternative.

In addition, a deprecation note should be added in the docstring, repeating the information
from the warning message:

```
.. deprecated:: 25.06
    Attribute `foo` was deprecated in version 25.06 and will be removed
    in 25.08. Use `metric` instead.
```

A deprecation requires a test which ensures that the warning is raised in relevant cases
but not in other cases. The warning should be caught in all other tests (using e.g., ``@pytest.mark.filterwarnings``).

### Public vs Private APIs
The following rules determine whether an API is considered public and therefore subject to the deprecation policy:

1. Any API prefixed with `_` is considered private and not subject to deprecation rules
2. APIs within the following namespaces are private and not subject to deprecation rules:
   - `internals`
   - `utils`
   - `common`
3. APIs within the `experimental` namespace are not subject to deprecation rules
4. For all other APIs, the determination of public vs private status is based on:
   - Presence in public documentation
   - Usage in public examples
   - Explicit declaration as part of the public API

When in doubt, treat an API as public to ensure proper deprecation cycles.

## Logging

cuML uses the [rapids-logger library](https://github.com/rapidsai/rapids-logger) for logging which in turn is built on top of [spdlog](https://github.com/gabime/spdlog). This provides fast, asynchronous logging with support for multiple sinks and formatting options.

### Usage

To emit log messages within the Python library, use the functions provided within the `cuml.internals.logger` module:

```python
from cuml.internals.logger import debug, info, warn, error, critical

# Example usage
debug("Detailed debug information")
info("General information")
warn("Warning message")
error("Error message")
critical("Critical error message")
```

### Log Levels

Use the appropriate log level based on the message's importance and target audience:

- **DEBUG**: Detailed information for debugging primarily aimed at developers
  - Use for detailed execution flow, variable values, and internal state
  - Example: "Initializing CUDA stream with device 0"

- **INFO**: General information about program execution primarily aimed at users
  - Use for messages that could be of interest to users, but usually don't require any further action
  - Example: "Building knn graph using nn descent"

- **WARNING**: Indicate a potential problem or other potentially surprising behavior
  - Use for ignored parameters, performance issues, or unexpected but handled conditions
  - Use for messages that we typically expect users to take action on
  - Example: "Parameter 'n_neighbors' was ignored as it's not supported in this mode"

- **ERROR**: Indicates a problem that prevents program execution
  - Use for recoverable errors that prevent the current operation
  - Example: "Failed to allocate GPU memory for input data"

- **CRITICAL**: A critical problem that is expected to lead to immediate program termination
  - Use sparingly, only for unrecoverable errors
  - Example: "CUDA device became unresponsive"

### Best Practices

1. **Message Content**
   - Be specific and include relevant context
   - Include error codes or exception details when applicable
   - Use consistent terminology

2. **Performance**
   - Avoid expensive string operations in debug messages
   - Do not emit log messages within loops that are input size dependent
   - Use lazy evaluation for debug messages:
   ```python
   from cuml.internals import logger

   if logger.should_log_for(logging.DEBUG):
       logger.debug(f"Expensive operation result: {expensive_operation()}")
   ```

3. **Formatting**
   - Use proper punctuation and capitalization
   - Keep messages concise but informative
   - Include relevant numeric values with units

4. **Context**
   - Include relevant parameters or state information
   - For errors, include the operation that failed
   - For warnings, explain why the behavior might be unexpected

### Configuration

The logging level can be configured using the `set_level` function from `cuml.internals.logger`:

```python
from cuml.internals import logger

# Set log level directly
logger.set_level(logger.level_enum.debug)

# Or use as a context manager for temporary level changes
with logger.set_level(logger.level_enum.debug):
    # Operations with debug logging
    pass
```

Available log levels:
- `logger.level_enum.debug`
- `logger.level_enum.info`
- `logger.level_enum.warn`
- `logger.level_enum.error`
- `logger.level_enum.critical`

## Multi-GPU Support

cuML provides limited multi-GPU support through Dask. Here is a basic example:

```python
import cudf
import dask_cudf
import numpy as np
from cuml.dask.linear_model import LogisticRegression
from dask.distributed import Client
from dask_cuda import LocalCUDACluster


def main():
    # Create sample data
    X = cudf.DataFrame(
        {
            "col1": np.array([1, 1, 2, 2], dtype=np.float32),
            "col2": np.array([1, 2, 2, 3], dtype=np.float32),
        }
    )
    y = cudf.Series(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32))

    # Convert to distributed dataframes
    X_ddf = dask_cudf.from_cudf(X, npartitions=2)
    y_ddf = dask_cudf.from_cudf(y, npartitions=2)

    # Train distributed model
    model = LogisticRegression()
    model.fit(X_ddf, y_ddf)

    # Make predictions
    prediction = model.predict(X_ddf)
    print(prediction.compute())


if __name__ == "__main__":
    # Create a local cluster with 2 GPUs
    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES="0,1")
    client = Client(cluster)
    main()
    client.close()
    cluster.close()
```

Key points for implementing multi-GPU estimators:

- Dask-based estimators should be implemented within the cuml.dask namespace
- The dask layer should focus on distributed computation, with base algorithms implemented in standard estimators
- See currently implemented estimators, e.g., LogisticRegression for examples on how to implement dask-based Multi-GPU estimators

## Benchmarking

The cuML code including its Python operations can be profiled. The `nvtx_benchmark.py` is a helper script that produces a simple benchmark summary. To use it, run `python nvtx_benchmark.py "python test.py"`.

Here is an example with the following script:
```python
from cuml.datasets import make_blobs
from cuml.manifold import UMAP

X, y = make_blobs(n_samples=1000, n_features=30)

model = UMAP()
model.fit(X)
embeddngs = model.transform(X)
```

that once benchmarked can have its profiling summarized:
```
datasets.make_blobs                                          :   1.3571 s

manifold.umap.fit [0x7f10eb69d4f0]                           :   0.6629 s
    |> umap::unsupervised::fit                               :   0.6611 s
    |==> umap::knnGraph                                      :   0.4693 s
    |==> umap::simplicial_set                                :   0.0015 s
    |==> umap::embedding                                     :   0.1902 s

manifold.umap.transform [0x7f10eb69d4f0]                     :   0.0934 s
    |> umap::transform                                       :   0.0925 s
    |==> umap::knnGraph                                      :   0.0909 s
    |==> umap::smooth_knn                                    :   0.0002 s
    |==> umap::optimization                                  :   0.0011 s
```
