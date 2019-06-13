# cuML 0.8.0 (Date TBD)

## New Features
- PR #636: Rand Index metric ml-prim
- PR #515: Added Random Projection feature
- PR #504: Contingency matrix ml-prim
- PR #612: Allow Cuda Array Interface, Numba inputs and input code refactor
- PR #641: C: Separate C-wrapper library build to generate libcuml.so
- PR #631: Add nvcategory based ordinal label encoder
- PR #670 Add test skipping functionality to build.sh

## Improvements

- PR #597: C++ cuML and ml-prims folder refactor
- PR #590: QN Recover from numeric errors
- PR #482: Introduce cumlHandle for pca and tsvd
- PR #573: Remove use of unnecessary cuDF column and series copies
- PR #601: Cython PEP8 cleanup and CI integration
- PR #596: Introduce cumlHandle for ols and ridge
- PR #579: Introduce cumlHandle for cd and sgd, and propagate C++ errors in cython level for cd and sgd
- PR #604: Adding cumlHandle to kNN, spectral methods, and UMAP
- PR #616: Enable clang-format for enforcing coding style
- PR #618: CI: Enable copyright header checks
- PR #622: Updated to use 0.8 dependencies
- PR #626: Added build.sh script, updated CI scripts and documentation
- PR #633: build: Auto-detection of GPU_ARCHS during cmake
- PR #650: Moving brute force kNN to prims. Creating stateless kNN API. 
- PR #662: C++: Bulk clang-format updates
- PR #671: Added pickle pytests and correct pickling of Base class
- PR #677: build: 'deep-clean' to build.sh to clean faiss build as well

## Bug Fixes
- PR #584: Added missing virtual destructor to deviceAllocator and hostAllocator
- PR #620: C++: Removed old unit-test files in ml-prims
- PR #627: C++: Fixed dbscan crash issue filed in 613
- PR #640: Remove setuptools from conda run dependency
- PR #646: Update link in contributing.md
- PR #649: Bug fix to LinAlg::reduce_rows_by_key prim filed in issue #648
- PR #666: fixes to gitutils.py to resolve both string decode and handling of uncommitted files
- PR #676: Fix template parameters in `bernoulli()` implementation.

# cuML 0.7.0 (10 May 2019)

## New Features

- PR #405: Quasi-Newton GLM Solvers
- PR #277: Add row- and column-wise weighted mean primitive
- PR #424: Add a grid-sync struct for inter-block synchronization
- PR #430: Add R-Squared Score to ml primitives
- PR #463: Add matrix gather to ml primitives
- PR #435: Expose cumlhandle in cython + developer guide
- PR #455: Remove default-stream arguement across ml-prims and cuML
- PR #375: cuml cpp shared library renamed to libcuml++.so
- PR #460: Random Forest & Decision Trees (Single-GPU, Classification)
- PR #491: Add doxygen build target for ml-prims
- PR #505: Add R-Squared Score to python interface
- PR #507: Add coordinate descent for lasso and elastic-net
- PR #511: Add a minmax ml-prim
- PR #516: Added Trustworthiness score feature
- PR #520: Add local build script to mimic gpuCI
- PR #503: Add column-wise matrix sort primitive
- PR #525: Add docs build script to cuML
- PR #528: Remove current KMeans and replace it with a new single GPU implementation built using ML primitives

## Improvements

- PR #481: Refactoring Quasi-Newton to use cumlHandle
- PR #467: Added validity check on cumlHandle_t
- PR #461: Rewrote permute and added column major version
- PR #440: README updates
- PR #295: Improve build-time and the interface e.g., enable bool-OutType, for distance()
- PR #390: Update docs version
- PR #272: Add stream parameters to cublas and cusolver wrapper functions
- PR #447: Added building and running mlprims tests to CI
- PR #445: Lower dbscan memory usage by computing adjacency matrix directly
- PR #431: Add support for fancy iterator input types to LinAlg::reduce_rows_by_key
- PR #394: Introducing cumlHandle API to dbscan and add example
- PR #500: Added CI check for black listed CUDA Runtime API calls
- PR #475: exposing cumlHandle for dbscan from python-side
- PR #395: Edited the CONTRIBUTING.md file
- PR #407: Test files to run stress, correctness and unit tests for cuml algos
- PR #512: generic copy method for copying buffers between device/host
- PR #533: Add cudatoolkit conda dependency
- PR #524: Use cmake find blas and find lapack to pass configure options to faiss
- PR #527: Added notes on UMAP differences from reference implementation
- PR #540: Use latest release version in update-version CI script
- PR #552: Re-enable assert in kmeans tests with xfail as needed
- PR #581: Add shared memory fast col major to row major function back with bound checks
- PR #592: More efficient matrix copy/reverse methods

## Bug Fixes

- PR #334: Fixed segfault in `ML::cumlHandle_impl::destroyResources`
- PR #349: Developer guide clarifications for cumlHandle and cumlHandle_impl
- PR #398: Fix CI scripts to allow nightlies to be uploaded
- PR #399: Skip PCA tests to allow CI to run with driver 418
- PR #422: Issue in the PCA tests was solved and CI can run with driver 418
- PR #409: Add entry to gitmodules to ignore build artifacts
- PR #412: Fix for svdQR function in ml-prims
- PR #438: Code that depended on FAISS was building everytime.
- PR #358: Fixed an issue when switching streams on MLCommon::device_buffer and MLCommon::host_buffer
- PR #434: Fixing bug in CSR tests
- PR #443: Remove defaults channel from ci scripts
- PR #384: 64b index arithmetic updates to the kernels inside ml-prims
- PR #459: Fix for runtime library path of pip package
- PR #464: Fix for C++11 destructor warning in qn
- PR #466: Add support for column-major in LinAlg::*Norm methods
- PR #465: Fixing deadlock issue in GridSync due to consecutive sync calls
- PR #468: Fix dbscan example build failure
- PR #470: Fix resource leakage in Kalman filter python wrapper
- PR #473: Fix gather ml-prim test for change in rng uniform API
- PR #477: Fixes default stream initialization in cumlHandle
- PR #480: Replaced qn_fit() declaration with #include of file containing definition to fix linker error
- PR #495: Update cuDF and RMM versions in GPU ci test scripts
- PR #499: DEVELOPER_GUIDE.md: fixed links and clarified ML::detail::streamSyncer example
- PR #506: Re enable ml-prim tests in CI
- PR #508: Fix for an error with default argument in LinAlg::meanSquaredError
- PR #519: README.md Updates and adding BUILD.md back
- PR #526: Fix the issue of wrong results when fit and transform of PCA are called separately
- PR #531: Fixing missing arguments in updateDevice() for RF
- PR #543: Exposing dbscan batch size through cython API and fixing broken batching
- PR #551: Made use of ZLIB_LIBRARIES consistent between ml_test and ml_mg_test
- PR #557: Modified CI script to run cuML tests before building mlprims and removed lapack flag
- PR #578: Updated Readme.md to add lasso and elastic-net
- PR #580: Fixing cython garbage collection bug in KNN
- PR #577: Use find libz in prims cmake
- PR #594: fixed cuda-memcheck mean_center test failures


# cuML 0.6.1 (09 Apr 2019)

## Bug Fixes

- PR #462 Runtime library path fix for cuML pip package


# cuML 0.6.0 (22 Mar 2019)

## New Features

- PR #249: Single GPU Stochastic Gradient Descent for linear regression, logistic regression, and linear svm with L1, L2, and elastic-net penalties.
- PR #247: Added "proper" CUDA API to cuML
- PR #235: NearestNeighbors MG Support
- PR #261: UMAP Algorithm
- PR #290: NearestNeighbors numpy MG Support
- PR #303: Reusable spectral embedding / clustering
- PR #325: Initial support for single process multi-GPU OLS and tSVD
- PR #271: Initial support for hyperparameter optimization with dask for many models

## Improvements

- PR #144: Dockerfile update and docs for LinearRegression and Kalman Filter.
- PR #168: Add /ci/gpu/build.sh file to cuML
- PR #167: Integrating full-n-final ml-prims repo inside cuml
- PR #198: (ml-prims) Removal of *MG calls + fixed a bug in permute method
- PR #194: Added new ml-prims for supporting LASSO regression.
- PR #114: Building faiss C++ api into libcuml
- PR #64: Using FAISS C++ API in cuML and exposing bindings through cython
- PR #208: Issue ml-common-3: Math.h: swap thrust::for_each with binaryOp,unaryOp
- PR #224: Improve doc strings for readable rendering with readthedocs
- PR #209: Simplify README.md, move build instructions to BUILD.md
- PR #218: Fix RNG to use given seed and adjust RNG test tolerances.
- PR #225: Support for generating random integers
- PR #215: Refactored LinAlg::norm to Stats::rowNorm and added Stats::colNorm
- PR #234: Support for custom output type and passing index value to main_op in *Reduction kernels
- PR #230: Refactored the cuda_utils header
- PR #236: Refactored cuml python package structure to be more sklearn like
- PR #232: Added reduce_rows_by_key
- PR #246: Support for 2 vectors in the matrix vector operator
- PR #244: Fix for single GPU OLS and Ridge to support one column training data
- PR #271: Added get_params and set_params functions for linear and ridge regression
- PR #253: Fix for issue #250-reduce_rows_by_key failed memcheck for small nkeys
- PR #269: LinearRegression, Ridge Python docs update and cleaning
- PR #322: set_params updated
- PR #237: Update build instructions
- PR #275: Kmeans use of faster gpu_matrix
- PR #288: Add n_neighbors to NearestNeighbors constructor
- PR #302: Added FutureWarning for deprecation of current kmeans algorithm
- PR #312: Last minute cleanup before release
- PR #315: Documentation updating and enhancements
- PR #330: Added ignored argument to pca.fit_transform to map to sklearn's implemenation
- PR #342: Change default ABI to ON
- PR #572: Pulling DBSCAN components into reusable primitives


## Bug Fixes

- PR #193: Fix AttributeError in PCA and TSVD
- PR #211: Fixing inconsistent use of proper batch size calculation in DBSCAN
- PR #202: Adding back ability for users to define their own BLAS
- PR #201: Pass CMAKE CUDA path to faiss/configure script
- PR #200 Avoid using numpy via cimport in KNN
- PR #228: Bug fix: LinAlg::unaryOp with 0-length input
- PR #279: Removing faiss-gpu references in README
- PR #321: Fix release script typo
- PR #327: Update conda requirements for version 0.6 requirements
- PR #352: Correctly calculating numpy chunk sizing for kNN
- PR #345: Run python import as part of package build to trigger compilation
- PR #347: Lowering memory usage of kNN.
- PR #355: Fixing issues with very large numpy inputs to SPMG OLS and tSVD.
- PR #357: Removing FAISS requirement from README
- PR #362: Fix for matVecOp crashing on large input sizes
- PR #366: Index arithmetic issue fix with TxN_t class
- PR #376: Disabled kmeans tests since they are currently too sensitive (see #71)
- PR #380: Allow arbitrary data size on ingress for numba_utils.row_matrix
- PR #385: Fix for long import cuml time in containers and fix for setup_pip
- PR #630: Fixing a missing kneighbors in nearest neighbors python proxy

# cuML 0.5.1 (05 Feb 2019)

## Bug Fixes

- PR #189 Avoid using numpy via cimport to prevent ABI issues in Cython compilation


# cuML 0.5.0 (28 Jan 2019)

## New Features

- PR #66: OLS Linear Regression
- PR #44: Distance calculation ML primitives
- PR #69: Ridge (L2 Regularized) Linear Regression
- PR #103: Linear Kalman Filter
- PR #117: Pip install support
- PR #64: Device to device support from cuML device pointers into FAISS

## Improvements

- PR #56: Make OpenMP optional for building
- PR #67: Github issue templates
- PR #44: Refactored DBSCAN to use ML primitives
- PR #91: Pytest cleanup and sklearn toyset datasets based pytests for kmeans and dbscan
- PR #75: C++ example to use kmeans
- PR #117: Use cmake extension to find any zlib installed in system
- PR #94: Add cmake flag to set ABI compatibility
- PR #139: Move thirdparty submodules to root and add symlinks to new locations
- PR #151: Replace TravisCI testing and conda pkg builds with gpuCI
- PR #164: Add numba kernel for faster column to row major transform
- PR #114: Adding FAISS to cuml build

## Bug Fixes

- PR #48: CUDA 10 compilation warnings fix
- PR #51: Fixes to Dockerfile and docs for new build system
- PR #72: Fixes for GCC 7
- PR #96: Fix for kmeans stack overflow with high number of clusters
- PR #105: Fix for AttributeError in kmeans fit method
- PR #113: Removed old  glm python/cython files
- PR #118: Fix for AttributeError in kmeans predict method
- PR #125: Remove randomized solver option from PCA python bindings


# cuML 0.4.0 (05 Dec 2018)

## New Features

## Improvements

- PR #42: New build system: separation of libcuml.so and cuml python package
- PR #43: Added changelog.md

## Bug Fixes


# cuML 0.3.0 (30 Nov 2018)

## New Features

- PR #33: Added ability to call cuML algorithms using numpy arrays

## Improvements

- PR #24: Fix references of python package from cuML to cuml and start using versioneer for better versioning
- PR #40: Added support for refactored cuDF 0.3.0, updated Conda files
- PR #33: Major python test cleaning, all tests pass with cuDF 0.2.0 and 0.3.0. Preparation for new build system
- PR #34: Updated batch count calculation logic in DBSCAN
- PR #35: Beginning of DBSCAN refactor to use cuML mlprims and general improvements

## Bug Fixes

- PR #30: Fixed batch size bug in DBSCAN that caused crash. Also fixed various locations for potential integer overflows
- PR #28: Fix readthedocs build documentation
- PR #29: Fix pytests for cuml name change from cuML
- PR #33: Fixed memory bug that would cause segmentation faults due to numba releasing memory before it was used. Also fixed row major/column major bugs for different algorithms
- PR #36: Fix kmeans gtest to use device data
- PR #38: cuda\_free bug removed that caused google tests to sometimes pass and sometimes fail randomly
- PR #39: Updated cmake to correctly link with CUDA libraries, add CUDA runtime linking and include source files in compile target

# cuML 0.2.0 (02 Nov 2018)

## New Features

- PR #11: Kmeans algorithm added
- PR #7: FAISS KNN wrapper added
- PR #21: Added Conda install support

## Improvements

- PR #15: Added compatibility with cuDF (from prior pyGDF)
- PR #13: Added FAISS to Dockerfile
- PR #21: Added TravisCI build system for CI and Conda builds

## Bug Fixes

- PR #4: Fixed explained variance bug in TSVD
- PR #5: Notebook bug fixes and updated results


# cuML 0.1.0

Initial release including PCA, TSVD, DBSCAN, ml-prims and cython wrappers
