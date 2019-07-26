# Defining cuML's Definition of Done Criteria


## Algorithm Completion Checklist

Below is a quick and simple checklist for developers to determine whether an algorithm is complete and ready for release. Most of these items contain more detailed descriptions in their corresponding developer guide. The checklist is broken down by layer (C++ or Python) and categorized further into

- **Design:** All algorithms should be designed with an eye on maintainability, performance, readability, and robustness.
- **Testing:** The goal for automated testing is to increase the code coverage as much as possible in order to ease time spent fixing bugs and developing new features. Additionally, a very important factor for a tool like `cuml` is to provide testing with multitple datasets that really stress the mathematical behavior of the algorithms, even if they don't increase code coverage! A comprehensive set of tests lowers the possibility for regressions and the introduction of bugs as the code evolves between versions. This covers both correctness & performance. 
- **Documentation:** User-facing documentation should be complete and descriptive. Developer-facing documentation should be used for constructs which are complex and/or not immediately obvious. 

### C++

#### Design

- Existing prims are used wherever possible
- Array inputs and outputs to algorithms are accepted on device
- New prims created wherever there is potential for reuse across different algorithms or prims
- User-facing API is stateless and follows the plain-old data (POD) design paradigm
- Public API contains a C-Wrapper around the stateless API
- Public API contains an Scikit-learn-like stateful wrapper around the stateless API

#### Testing

- Prims: GTests with different inputs
- Algorithms: End-to-end GTests with different inputs and different datasets

#### Documentation

- Complete and comprehensive Doxygen strings explaining the public API, restrictions, and gotchas. Any array parameters should also note whether the underlying memory is host or device. 


### Python

#### Design

- Python class is as "near drop-in replacement" for Scikit-learn (or relevant industry standard) API as possible
    - It is recommended to open an initial PR with the API design if there are going to be significant differences with reference APIs, or lack of a reference API, to have a discussion about it. 
- Python class is pickleable
- APIs use `input_to_dev_array` to accept flexible inputs and check their datatypes

#### Testing 

- Pytests for wrapper functionality against Scikit-learn using relevant datasets
- Stress tests against reasonable inputs (e.g short-wide, tall-narrow, different numerical precision)
- Pytests for pickle capability
- Pytests to evaluate correctness against Scikit-learn on a variety of datasets
- Add algorithm to benchmarks packages

#### Documentation

- Complete and comprehensive Pydoc strings explaining public API, restrictions, and gotchas
- Docstrings include references to the underlying algorithm (e.g paper or Scikit-learn algorithm being implemented or a description of the algorithm used if nonstandard).


## Algorithm Status

Sometimes it is possible that only some of these requirements are satisfied before an algorithm gets released. We assign a status of `bronze`, `silver`, or `gold` to denote the amount of work necessary to provide the most robust, performant, and maintainable algorithm possible.

Note that a change to an algorithm's classification can go up or down, depending on circumstances. For example, an algorithm that has been considered `Gold status` might drop to a `Silver status` when being updated to a new low-level feature or library. 

### Bronze: 
- Known design flaws affecting sustainability and/or performance require a significant effort / redesign / refactor
- C++ checklist / API functionality is complete but Python API has not been built
- ure Python API exists without using C++ for processing when a C++ implementation would provide a significant performance boost
- There are known issues that affect memory / accuracy / stability on some inputs
- Coverage of automated tests in C++ or Python is minimal and could be significantly improved
- Outdated / legacy code that requires significant work and refactoring


### Silver:
- Known design flaws affecting sustainability and/or performance are fairly low in complexity (can be fixed quickly and without too much effort)
- Automated tests in C++ or Python are not as extensive as they could be regarding different datasets / inputs, but they do test general functionality
- There are known (or potential) bugs that affect memory / accuracy, but only on a small subset of specific inputs
- There are known optimizations that would provide a significant performance boost over the current implementation


### Gold: 
- Design is robust, maintainable, and performant without any known flaws.
- Algorithm is as performant as possible without any significant known or potential optimizations over the current implementation.
- Algorithm is consistently correct and bug free on all datasets.
- There are no known memory usage limitations resulting from the design choices that could be reasonably refactored.


## Review Checklist

Aside from the general algorithm expectations outlined in the checklists above, code reviewers should use the following checklist to make sure the algorithm meets cuML standards. 

### All

- New files contain necessary license headers
- Diff does not contain files with excess formatting changes, without other changes also being made to the file
- Code does not contain any immediately obvious memory leaks or garbage collection issues 

### C++

- New GTests are being enabled in `CMakeLists.txt`


