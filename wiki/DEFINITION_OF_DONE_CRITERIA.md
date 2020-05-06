# Defining cuML's Definition of Done Criteria


## Algorithm Completion Checklist

Below is a quick and simple checklist for developers to determine whether an algorithm is complete and ready for release. Most of these items contain more detailed descriptions in their corresponding developer guide. The checklist is broken down by layer (C++ or Python) and categorized further into

- **Design:** All algorithms should be designed with an eye on maintainability, performance, readability, and robustness.
- **Testing:** The goal for automated testing is to increase both the spread and the depth of code coverage as much as possible in order to ease time spent fixing bugs and developing new features. Additionally, a very important factor for a tool like `cuml` is to provide testing with multiple datasets that really stress the mathematical behavior of the algorithms. A comprehensive set of tests lowers the possibility for regressions and the introduction of bugs as the code evolves between versions. This covers both correctness & performance. 
- **Documentation:** User-facing documentation should be complete and descriptive. Developer-facing documentation should be used for constructs which are complex and/or not immediately obvious. 
- **Performance:** Algorithms should be [benchmarked] and profiled regularly to spot potential bottlenecks, performance regressions, and memory problems.

### C++

#### Design

- Existing prims are used wherever possible
- Array inputs and outputs to algorithms are accepted on device
- New prims created wherever there is potential for reuse across different algorithms or prims
- User-facing API is [stateless](cpp/DEVELOPER_GUIDE.md#public-cuml-interface) and follows the [plain-old data (POD)](https://en.wikipedia.org/wiki/Passive_data_structure) design paradigm
- Public API contains a C-Wrapper around the stateless API
- (optional) Public API contains an Scikit-learn-like stateful wrapper around the stateless API

#### Testing

- Prims: GTests with different inputs
- Algorithms: End-to-end GTests with different inputs and different datasets

#### Documentation

- Complete and comprehensive [Doxygen](http://www.doxygen.nl/manual/docblocks.html) strings explaining the public API, restrictions, and gotchas. Any array parameters should also note whether the underlying memory is host or device.
- Array inputs/outputs should also mention their expected size/dimension.
- If there are references to the underlying algorithm, they must be cited too.


### Python

#### Design

- Python class is as "near drop-in replacement" for Scikit-learn (or relevant industry standard) API as possible. This means parameters have the same names as Scikit-learn, and where differences exist, they are clearly documented in docstrings.
- It is recommended to open an initial PR with the API design if there are going to be significant differences with reference APIs, or lack of a reference API, to have a discussion about it. 
- Python class is pickleable and a test has been added to `cuml/test/test_pickle.py`
- APIs use `input_to_cuml_array` to accept flexible inputs and check their datatypes and use `cumlArray.to_output()` to return configurable outputs.
- Any internal parameters or array-based instance variables use `CumlArray`

#### Testing 

- Pytests for wrapper functionality against Scikit-learn using relevant datasets
- Stress tests against reasonable inputs (e.g short-wide, tall-narrow, different numerical precision)
- Pytests for pickle capability
- Pytests to evaluate correctness against Scikit-learn on a variety of datasets
- Add algorithm to benchmarks package in `python/cuml/benchmarks/algorithms.py` and benchmarks notebook in `python/cuml/notebooks/tools/cuml_benchmarks.ipynb`

#### Documentation

- Complete and comprehensive Pydoc strings explaining public API, restrictions, a usage example, and gotchas. This should be in [Numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) format
- Docstrings include references to any scientific papers or standard publications on the underlying algorithm (e.g paper or Scikit-learn algorithm being implemented or a description of the algorithm used if nonstandard).

## Algorithm Status

Sometimes it is possible that only some of these requirements are satisfied before an algorithm gets released. We are beginning to assign a status of `bronze`, `silver`, or `gold` to denote an algorithm's level of completeness and robustness. Algorithms which do not meet the minimal criteria for the `bronze` status are assigned the status of `unclassified`. 

Note that a change to an algorithm's classification can go up or down, depending on circumstances. For example, an algorithm that has been considered `Gold status` might drop to a `Silver status` when being updated to a new low-level feature or library. 

### Unclassified:
- C++ checklist / API functionality is complete but Python API has not been built
- There are known issues significantly affecting the accuracy or user's confidence in correct outputs
- Outdated, stale, or legacy code which is either known or evidenced to longer be in active use
- Documentation is lacking or non-existent
- Either no profiling or benchmarking have been conducted or they have been found to provide only a marginal benefit


### Bronze: 
- Known design flaws affecting sustainability, maintainability and/or performance are non-trivial
- Issues affecting memory, stability, or unexpected behavior on some inputs are known to have a negative impact on the needs of users
- Coverage of automated tests in C++ or Python is insufficient and could be significantly improved
- Some features might exist, but implementation is overall incomplete and/or lacking core features that impact its utility
- Algorithm has been benchmarked and found to have some speedup, but implementation could be significantly improved
- Documentation exists, but could be significantly improved


### Silver:
- Known design flaws affecting sustainability and/or performance are fairly minor.
- Automated tests in C++ or Python are not as extensive as they should be regarding different datasets / inputs, but they do test the general features
- There are known (or potential) bugs that affect memory / accuracy, but only on a small subset of specific inputs
- Overall, performance gains are acceptable, but there are known optimizations that would provide a significant increase in performance over the the existing implementation
- Documentation is mostly complete, but not comprehensive


### Gold: 
- Design is robust, maintainable, and performant without any known significant flaws in either of the categories aforementioned.
- Algorithm demonstrates consistent speedups over CPU comparison and has no major known bottlenecks
- Algorithm is consistently correct and bug free on all tested datasets.
- There are no known memory usage limitations resulting from the design choices that could be reasonably refactored.
- Algorithm has found use in pipelines and been evaluated by users outside of the cuml codebase, and the team of regular cuml contributors.
- Documentation is comprehensive


## Review Checklist

Aside from the general algorithm expectations outlined in the checklists above, code reviewers should use the following checklist to make sure the algorithm meets cuML standards. 

### All

- New files contain necessary license headers
- Diff does not contain files with excess formatting changes, without other changes also being made to the file
- Code does not contain any known serious memory leaks or garbage collection issues
- Modifications are cohesive and in-scope for the PR's intended purpose
- Changes to the public API will not have a negative impact to existing users between minor versions (eg. large changes to very popular public APIs go through a deprecation cycle to preserve backwards compatibility)
- Where it is reasonable to do so, unexpected inputs fail gracefully and provide actionable feedback to the user
- Automated tests properly exercise the changes in the PR
- New algorithms provide benchmarks (both C++ and Python) 


### C++

- New GTests are being enabled in `CMakeLists.txt`

