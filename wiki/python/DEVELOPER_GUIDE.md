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
We use [https://docs.pytest.org/en/latest/]() for writing and running tests. To see existing examples, refer to any of the `test_*.py` files in the folder `cuml/test`.

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
