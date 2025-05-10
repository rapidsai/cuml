# cuML Python Developer Guide
This document summarizes guidelines and best practices for contributions to the python component of the library cuML, the machine learning component of the RAPIDS ecosystem. This is an evolving document so contributions, clarifications and issue reports are highly welcome.

## General
Please start by reading:
1. [CONTRIBUTING.md](../../CONTRIBUTING.md).
2. [C++ DEVELOPER_GUIDE.md](../cpp/DEVELOPER_GUIDE.md)
3. [Python cuML README.md](../../python/README.md)

## Thread safety
Refer to the section on thread safety in [C++ DEVELOPER_GUIDE.md](../cpp/DEVELOPER_GUIDE.md#thread-safety)

## Coding style
1. [PEP8](https://www.python.org/dev/peps/pep-0008) and [flake8](http://flake8.pycqa.org/en/latest/) is used to check the adherence to this style.
2. [sklearn coding guidelines](https://scikit-learn.org/stable/developers/contributing.html#coding-guidelines)

## Creating class for a new estimator or other ML algorithm
1. Make sure that this algo has been implemented in the C++ side. Refer to [C++ DEVELOPER_GUIDE.md](../cpp/DEVELOPER_GUIDE.md) for guidelines on developing in C++.
2. Refer to the [next section](DEVELOPER_GUIDE.md#creating-python-wrapper-class-for-an-existing-ml-algo) for the remaining steps.

## Creating python estimator wrapper class
1. Create a corresponding algoName.pyx file inside `python/cuml` folder.
2. Ensure that the folder structure inside here reflects that of sklearn's. Example, `pca.pyx` should be kept inside the `decomposition` sub-folder of `python/cuml`.
.  Match the corresponding scikit-learn's interface as closely as possible. Refer to their [developer guide](https://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects) on API design of sklearn objects for details.
3. Always make sure to have your class inherit from `cuml.Base` class as your parent/ancestor.
4. Ensure that the estimator's output fields follow the 'underscore on both sides' convention explained in the documentation of `cuml.Base`. This allows it to support configurable output types.

For an in-depth guide to creating estimators, see the [Estimator Guide](ESTIMATOR_GUIDE.md)

## Error handling
If you are trying to call into cuda runtime APIs inside `cuml.cuda`, in case of any errors, they'll raise a `cuml.cuda.CudaRuntimeError`. For example:
```python
from cuml.cuda import Stream, CudaRuntimeError
try:
    s = Stream()
    s.sync
except CudaRuntimeError as cre:
    print("Cuda Error! '%s'" % str(cre))
```

## Logging
TBD

## Documentation
We mostly follow [PEP 257](https://www.python.org/dev/peps/pep-0257/) style docstrings for documenting the interfaces.
The examples in the documentation are checked through doctest. To skip the check for an example's output, use the command `# doctest: +SKIP`.
Examples subject to numerical imprecision, or that can't be reproduced consistently should be skipped.

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

## Device and Host memory allocations
TODO: talk about enabling RMM here when it is ready

## Asynchronous operations and stream ordering
If you want to schedule the execution of two algorithms concurrently, it is better to create two separate streams and assign them to separate handles. Finally, schedule the algorithms using these handles.
```python
import cuml
from cuml.cuda import Stream
s1 = Stream()
h1 = cuml.Handle()
h1.setStream(s1)
s2 = Stream()
h2 = cuml.Handle()
h2.setStream(s2)
algo1 = cuml.Algo1(handle=h1, ...)
algo2 = cuml.Algo2(handle=h2, ...)
algo1.fit(X1, y1)
algo2.fit(X2, y2)
```
To know more underlying details about stream ordering refer to the corresponding section of [C++ DEVELOPER_GUIDE.md](../../cpp/DEVELOPER_GUIDE.md#asynchronous-operations-and-stream-ordering)

## Multi GPU

TODO: Add more details.

## Benchmarking

The cuML code including its Python operations can be profiled. The `nvtx_benchmark.py` is a helper script that produces a simple benchmark summary. To use it, run `python nvtx_benchmark.py "python test.py"`.

Here is an example with the following script:
```
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
