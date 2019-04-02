# cuML Python Developer Guide
This document summarizes guidelines and best practices for contributions to the python component of the library cuML, the machine learning component of the RAPIDS ecosystem. This is an evolving document so contributions, clarifications and issue reports are highly welcome.

## General
Please start by reading:
1. [CONTRIBUTING.md](../CONTRIBUTING.md).
2. [C++ DEVELOPER_GUIDE.md](../cuML/DEVELOPER_GUIDE.md)
3. [Python cuML README.md](README.md)

## Thread safety
Refer to the section on thread safety in [C++ DEVELOPER_GUIDE.md](../cuML/DEVELOPER_GUIDE.md#thread-safety)

## Coding style
1. [PEP8](https://www.python.org/dev/peps/pep-0008) and [flake8](http://flake8.pycqa.org/en/latest/) is used to check the adherence to this style.
2. [sklearn coding guidelines](https://scikit-learn.org/stable/developers/contributing.html#coding-guidelines)

## Creating class for a new ML algo
1. Make sure that this algo has been implemented in the C++ side. Refer to [C++ DEVELOPER_GUIDE.md](../cuML/DEVELOPER_GUIDE.md) for guidelines on developing in C++.
2. Refer to the [next section](./DEVELOPER_GUIDE.md#creating-python-wrapper-class-for-an-existing-ml-algo) for the remaining steps.

## Creating python wrapper class for an existing ML algo
1. Create a corresponding algoName.pyx file inside `python/cuml` folder.
2. Note that the folder structure inside here should reflect that of sklearn's. Example, `pca.pyx` should be kept inside the `decomposition` sub-folder of `python/cuml`.
. We try to match the corresponding scikit-learn's interface as closely as possible. Refer to their [developer guide](https://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects) on API design of sklearn objects for details.
3. Always make sure to have your class inherit from `cuml.common.base.Base` class as your parent/ancestor.

## Error handling
If you are trying to call into cuda runtime APIs inside `cuml.common.cuda`, in case of any errors, they'll raise a `cuml.common.cuda.CudaRuntimeError`. For example:
```python
from cuml.common.cuda import Stream, CudaRuntimeError
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

## Testing and Unit Testing
We use [https://docs.pytest.org/en/latest/]() for writing and running tests. To see existing examples, refer to any of the `test_*.py` files in the folder `cuml/test`.

## Device and Host memory allocations
TODO: talk about enabling RMM here when it is ready

## Asynchronous operations and stream ordering
If you want to schedule the execution of two algorithms concurrently, it is better to create two separate streams and assign them to separate handles. Finally, schedule the algorithms using these handles.
```python
from cuml.common.cuda import Stream
from cuml.common.handle import Handle
s1 = Stream()
h1 = Handle()
h1.setStream(s1)
s2 = Stream()
h2 = Handle()
h2.setStream(s2)
algo1 = cuml.Algo1(handle=h1, ...)
algo2 = cuml.Algo2(handle=h2, ...)
algo1.fit(X1, y1)
algo2.fit(X2, y2)
```
To know more underlying details about stream ordering refer to the corresponding section of [C++ DEVELOPER_GUIDE.md](../cuML/DEVELOPER_GUIDE.md#asynchronous-operations-and-stream-ordering)

## Multi GPU
We currently have **S**ingle **P**rocess **M**ultiple **G**PU (SPMG) versions of KNN, OLS and tSVD. Our upcoming versions will concentrate on **O**ne **P**rocess per **G**PU (OPG) paradigm.

TODO: Add more details.
