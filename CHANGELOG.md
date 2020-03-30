# cuML 0.13.0 (31 Mar 2020)

## New Features
- PR #1777: Python bindings for entropy
- PR #1742: Mean squared error implementation with cupy
- PR #1766: Mean absolute error implementation with cupy
- PR #1766: Mean squared log error implementation with cupy
- PR #1635: cuML Array shim and configurable output added to cluster methods
- PR #1586: Seasonal ARIMA
- PR #1683: cuml.dask make_regression
- PR #1689: Add framework for cuML Dask serializers
- PR #1709: Add `decision_function()` and `predict_proba()` for LogisticRegression
- PR #1714: Add `print_env.sh` file to gather important environment details
- PR #1750: LinearRegression CumlArray for configurable output
- PR #1767: Single GPU decomposition models configurable output
- PR #1646: Using FIL to predict in MNMG RF
- PR #1778: Make cuML Handle picklable
- PR #1738: cuml.dask refactor beginning and dask array input option for OLS, Ridge and KMeans
- PR #1874: Add predict_proba function to RF classifier
- PR #1815: Adding KNN parameter to UMAP

## Improvements
- PR #1644: Add `predict_proba()` for FIL binary classifier
- PR #1620: Pickling tests now automatically finds all model classes inheriting from cuml.Base
- PR #1637: Update to newer treelite version with XGBoost 1.0 compatibility
- PR #1632: Fix MBSGD models inheritance, they now inherits from cuml.Base
- PR #1628: Remove submodules from cuML
- PR #1755: Expose the build_treelite function for python
- PR #1649: Add the fil_sparse_format variable option to RF API
- PR #1647: storage_type=AUTO uses SPARSE for large models
- PR #1668: Update the warning statement thrown in RF when the seed is set but n_streams is not 1
- PR #1662: use of direct cusparse calls for coo2csr, instead of depending on nvgraph
- PR #1747: C++: dbscan performance improvements and cleanup
- PR #1697: Making trustworthiness batchable and using proper workspace
- PR #1721: Improving UMAP pytests
- PR #1717: Call `rmm_cupy_allocator` for CuPy allocations
- PR #1718: Import `using_allocator` from `cupy.cuda`
- PR #1723: Update RF Classifier to throw an exception for multi-class pickling
- PR #1726: Decorator to allocate CuPy arrays with RMM
- PR #1719: UMAP random seed reproducibility
- PR #1748: Test serializing `CumlArray` objects
- PR #1776: Refactoring pca/tsvd distributed
- PR #1762: Update CuPy requirement to 7
- PR #1768: C++: Different input and output types for add and subtract prims
- PR #1790: Add support for multiple seeding in k-means++
- PR #1805: Adding new Dask cuda serializers to naive bayes + a trivial perf update
- PR #1812: C++: bench: UMAP benchmark cases added
- PR #1795: Add capability to build CumlArray from bytearray/memoryview objects
- PR #1824: C++: improving the performance of UMAP algo
- PR #1816: Add ARIMA notebook
- PR #1856: Update docs for 0.13
- PR #1827: Add HPO demo Notebook
- PR #1825: `--nvtx` option in `build.sh`
- PR #1847: Update XGBoost version for CI
- PR #1837: Simplify cuML Array construction
- PR #1848: Rely on subclassing for cuML Array serialization
- PR #1866: Minimizing client memory pressure on Naive Bayes
- PR #1788: Removing complexity bottleneck in S-ARIMA
- PR #1891: Additional improvements to naive bayes tree reduction

## Bug Fixes
- PR #1835 : Fix calling default RF Classification always
- PT #1904: replace cub sort
- PR #1833: Fix depth issue in shallow RF regression estimators
- PR #1770: Warn that KalmanFilter is deprecated
- PR #1775: Allow CumlArray to work with inputs that have no 'strides' in array interface
- PR #1594: Train-test split is now reproducible
- PR #1590: Fix destination directory structure for run-clang-format.py
- PR #1611: Fixing pickling errors for KNN classifier and regressor
- PR #1617: Fixing pickling issues for SVC and SVR
- PR #1634: Fix title in KNN docs
- PR #1627: Adding a check for multi-class data in RF classification
- PR #1654: Skip treelite patch if its already been applied
- PR #1661: Fix nvstring variable name
- PR #1673: Using struct for caching dlsym state in communicator
- PR #1659: TSNE - introduce 'convert_dtype' and refactor class attr 'Y' to 'embedding_'
- PR #1672: Solver 'svd' in Linear and Ridge Regressors when n_cols=1
- PR #1670: Lasso & ElasticNet - cuml Handle added
- PR #1671: Update for accessing cuDF Series pointer
- PR #1652: Support XGBoost 1.0+ models in FIL
- PR #1702: Fix LightGBM-FIL validation test
- PR #1701: test_score kmeans test passing with newer cupy version
- PR #1706: Remove multi-class bug from QuasiNewton
- PR #1699: Limit CuPy to <7.2 temporarily
- PR #1708: Correctly deallocate cuML handles in Cython
- PR #1730: Fixes to KF for test stability (mainly in CUDA 10.2)
- PR #1729: Fixing naive bayes UCX serialization problem in fit()
- PR #1749: bug fix rf classifier/regressor on seg fault in bench
- PR #1751: Updated RF documentation
- PR #1765: Update the checks for using RF GPU predict
- PR #1787: C++: unit-tests to check for RF accuracy. As well as a bug fix to improve RF accuracy
- PR #1793: Updated fil pyx to solve memory leakage issue
- PR #1810: Quickfix - chunkage in dask make_regression
- PR #1842: DistributedDataHandler not properly setting 'multiple'
- PR #1849: Critical fix in ARIMA initial estimate
- PR #1851: Fix for cuDF behavior change for multidimensional arrays
- PR #1852: Remove Thrust warnings
- PR #1868: Turning off IPC caching until it is fixed in UCX-py/UCX
- PR #1876: UMAP exponential decay parameters fix
- PR #1887: Fix hasattr for missing attributes on base models
- PR #1877: Remove resetting index in shuffling in train_test_split
- PR #1893: Updating UCX in comms to match current UCX-py
- PR #1888: Small train_test_split test fix
- PR #1899: Fix dask `extract_partitions()`, remove transformation as instance variable in PCA and TSVD and match sklearn APIs
- PR #1920: Temporarily raising threshold for UMAP reproducibility tests
- PR #1918: Create memleak fixture to skip memleak tests in CI for now
- PR #1926: Update batch matrix test margins
- PR #1925: Fix failing dask tests
- PR #1932: Isolating cause of make_blobs failure
- PR #1951: Dask Random forest regression CPU predict bug fix
- PR #1948: Adjust BatchedMargin margin and disable tests temporarily


# cuML 0.12.0 (04 Feb 2020)

## New Features
- PR #1483: prims: Fused L2 distance and nearest-neighbor prim
- PR #1494: bench: ml-prims benchmark
- PR #1514: bench: Fused L2 NN prim benchmark
- PR #1411: Cython side of MNMG OLS
- PR #1520: Cython side of MNMG Ridge Regression
- PR #1516: Suppor Vector Regression (epsilon-SVR)

## Improvements
- PR #1638: Update cuml/docs/README.md
- PR #1468: C++: updates to clang format flow to make it more usable among devs
- PR #1473: C++: lazy initialization of "costly" resources inside cumlHandle
- PR #1443: Added a new overloaded GEMM primitive
- PR #1489: Enabling deep trees using Gather tree builder
- PR #1463: Update FAISS submodule to 1.6.1
- PR #1488: Add codeowners
- PR #1432: Row-major (C-style) GPU arrays for benchmarks
- PR #1490: Use dask master instead of conda package for testing
- PR #1375: Naive Bayes & Distributed Naive Bayes
- PR #1377: Add GPU array support for FIL benchmarking
- PR #1493: kmeans: add tiling support for 1-NN computation and use fusedL2-1NN prim for L2 distance metric
- PR #1532: Update CuPy to >= 6.6 and allow 7.0
- PR #1528: Re-enabling KNN using dynamic library loading for UCX in communicator
- PR #1545: Add conda environment version updates to ci script
- PR #1541: Updates for libcudf++ Python refactor
- PR #1555: FIL-SKL, an SKLearn-based benchmark for FIL
- PR #1537: Improve pickling and scoring suppport for many models to support hyperopt
- PR #1551: Change custom kernel to cupy for col/row order transform
- PR #1533: C++: interface header file separation for SVM
- PR #1560: Helper function to allocate all new CuPy arrays with RMM memory management
- PR #1570: Relax nccl in conda recipes to >=2.4 (matching CI)
- PR #1578: Add missing function information to the cuML documenataion
- PR #1584: Add has_scipy utility function for runtime check
- PR #1583: API docs updates for 0.12
- PR #1591: Updated FIL documentation

## Bug Fixes
- PR #1470: Documentation: add make_regression, fix ARIMA section
- PR #1482: Updated the code to remove sklearn from the mbsgd stress test
- PR #1491: Update dev environments for 0.12
- PR #1512: Updating setup_cpu() in SpeedupComparisonRunner
- PR #1498: Add build.sh to code owners
- PR #1505: cmake: added correct dependencies for prims-bench build
- PR #1534: Removed TODO comment in create_ucp_listeners()
- PR #1548: Fixing umap extra unary op in knn graph
- PR #1547: Fixing MNMG kmeans score. Fixing UMAP pickling before fit(). Fixing UMAP test failures.
- PR #1557: Increasing threshold for kmeans score
- PR #1562: Increasing threshold even higher
- PR #1564: Fixed a typo in function cumlMPICommunicator_impl::syncStream
- PR #1569: Remove Scikit-learn exception and depedenncy in SVM
- PR #1575: Add missing dtype parameter in call to strides to order for CuPy 6.6 code path
- PR #1574: Updated the init file to include SVM
- PR #1589: Fixing the default value for RF and updating mnmg predict to accept cudf
- PR #1601: Fixed wrong datatype used in knn voting kernel

# cuML 0.11.0 (11 Dec 2019)

## New Features

- PR #1295: Cython side of MNMG PCA
- PR #1218: prims: histogram prim
- PR #1129: C++: Separate include folder for C++ API distribution
- PR #1282: OPG KNN MNMG Code (disabled for 0.11)
- PR #1242: Initial implementation of FIL sparse forests
- PR #1194: Initial ARIMA time-series modeling support.
- PR #1286: Importing treelite models as FIL sparse forests
- PR #1285: Fea minimum impurity decrease RF param
- PR #1301: Add make_regression to generate regression datasets
- PR #1322: RF pickling using treelite, protobuf and FIL
- PR #1332: Add option to cuml.dask make_blobs to produce dask array
- PR #1307: Add RF regression benchmark
- PR #1327: Update the code to build treelite with protobuf
- PR #1289: Add Python benchmarking support for FIL
- PR #1371: Cython side of MNMG tSVD
- PR #1386: Expose SVC decision function value

## Improvements
- PR #1170: Use git to clone subprojects instead of git submodules
- PR #1239: Updated the treelite version
- PR #1225: setup.py clone dependencies like cmake and correct include paths
- PR #1224: Refactored FIL to prepare for sparse trees
- PR #1249: Include libcuml.so C API in installed targets
- PR #1259: Conda dev environment updates and use libcumlprims current version in CI
- PR #1277: Change dependency order in cmake for better printing at compile time
- PR #1264: Add -s flag to GPU CI pytest for better error printing
- PR #1271: Updated the Ridge regression documentation
- PR #1283: Updated the cuMl docs to include MBSGD and adjusted_rand_score
- PR #1300: Lowercase parameter versions for FIL algorithms
- PR #1312: Update CuPy to version 6.5 and use conda-forge channel
- PR #1336: Import SciKit-Learn models into FIL
- PR #1314: Added options needed for ASVDb output (CUDA ver, etc.), added option
  to select algos
- PR #1335: Options to print available algorithms and datasets
  in the Python benchmark
- PR #1338: Remove BUILD_ABI references in CI scripts
- PR #1340: Updated unit tests to uses larger dataset
- PR #1351: Build treelite temporarily for GPU CI testing of FIL Scikit-learn
  model importing
- PR #1367: --test-split benchmark parameter for train-test split
- PR #1360: Improved tests for importing SciKit-Learn models into FIL
- PR #1368: Add --num-rows benchmark command line argument
- PR #1351: Build treelite temporarily for GPU CI testing of FIL Scikit-learn model importing
- PR #1366: Modify train_test_split to use CuPy and accept device arrays
- PR #1258: Documenting new MPI communicator for multi-node multi-GPU testing
- PR #1345: Removing deprecated should_downcast argument
- PR #1362: device_buffer in UMAP + Sparse prims
- PR #1376: AUTO value for FIL algorithm
- PR #1408: Updated pickle tests to delete the pre-pickled model to prevent pointer leakage
- PR #1357: Run benchmarks multiple times for CI
- PR #1382: ARIMA optimization: move functions to C++ side
- PR #1392: Updated RF code to reduce duplication of the code
- PR #1444: UCX listener running in its own isolated thread
- PR #1445: Improved performance of FIL sparse trees
- PR #1431: Updated API docs
- PR #1441: Remove unused CUDA conda labels
- PR #1439: Match sklearn 0.22 default n_estimators for RF and fix test errors
- PR #1461: Add kneighbors to API docs

## Bug Fixes
- PR #1281: Making rng.h threadsafe
- PR #1212: Fix cmake git cloning always running configure in subprojects
- PR #1261: Fix comms build errors due to cuml++ include folder changes
- PR #1267: Update build.sh for recent change of building comms in main CMakeLists
- PR #1278: Removed incorrect overloaded instance of eigJacobi
- PR #1302: Updates for numba 0.46
- PR #1313: Updated the RF tests to set the seed and n_streams
- PR #1319: Using machineName arg passed in instead of default for ASV reporting
- PR #1326: Fix illegal memory access in make_regression (bounds issue)
- PR #1330: Fix C++ unit test utils for better handling of differences near zero
- PR #1342: Fix to prevent memory leakage in Lasso and ElasticNet
- PR #1337: Fix k-means init from preset cluster centers
- PR #1354: Fix SVM gamma=scale implementation
- PR #1344: Change other solver based methods to create solver object in init
- PR #1373: Fixing a few small bugs in make_blobs and adding asserts to pytests
- PR #1361: Improve SMO error handling
- PR #1384: Lower expectations on batched matrix tests to prevent CI failures
- PR #1380: Fix memory leaks in ARIMA
- PR #1391: Lower expectations on batched matrix tests even more
- PR #1394: Warning added in svd for cuda version 10.1
- PR #1407: Resolved RF predict issues and updated RF docstring
- PR #1401: Patch for lbfgs solver for logistic regression with no l1 penalty
- PR #1416: train_test_split numba and rmm device_array output bugfix
- PR #1419: UMAP pickle tests are using wrong n_neighbors value for trustworthiness
- PR #1438: KNN Classifier to properly return Dataframe with Dataframe input
- PR #1425: Deprecate seed and use random_state similar to Scikit-learn in train_test_split
- PR #1458: Add joblib as an explicit requirement
- PR #1474: Defer knn mnmg to 0.12 nightly builds and disable ucx-py dependency

# cuML 0.10.0 (16 Oct 2019)

## New Features
- PR #1148: C++ benchmark tool for c++/CUDA code inside cuML
- PR #1071: Selective eigen solver of cuSolver
- PR #1073: Updating RF wrappers to use FIL for GPU accelerated prediction
- PR #1104: CUDA 10.1 support
- PR #1113: prims: new batched make-symmetric-matrix primitive
- PR #1112: prims: new batched-gemv primitive
- PR #855: Added benchmark tools
- PR #1149 Add YYMMDD to version tag for nightly conda packages
- PR #892: General Gram matrices prim
- PR #912: Support Vector Machine
- PR #1274: Updated the RF score function to use GPU predict

## Improvements
- PR #961: High Peformance RF; HIST algo
- PR #1028: Dockerfile updates after dir restructure. Conda env yaml to add statsmodels as a dependency
- PR #1047: Consistent OPG interface for kmeans, based on internal libcumlprims update
- PR #763: Add examples to train_test_split documentation
- PR #1093: Unified inference kernels for different FIL algorithms
- PR #1076: Paying off some UMAP / Spectral tech debt.
- PR #1086: Ensure RegressorMixin scorer uses device arrays
- PR #1110: Adding tests to use default values of parameters of the models
- PR #1108: input_to_host_array function in input_utils for input processing to host arrays
- PR #1114: K-means: Exposing useful params, removing unused params, proxying params in Dask
- PR #1138: Implementing ANY_RANK semantics on irecv
- PR #1142: prims: expose separate InType and OutType for unaryOp and binaryOp
- PR #1115: Moving dask_make_blobs to cuml.dask.datasets. Adding conversion to dask.DataFrame
- PR #1136: CUDA 10.1 CI updates
- PR #1135: K-means: add boundary cases for kmeans||, support finer control with convergence
- PR #1163: Some more correctness improvements. Better verbose printing
- PR #1165: Adding except + in all remaining cython
- PR #1186: Using LocalCUDACluster Pytest fixture
- PR #1173: Docs: Barnes Hut TSNE documentation
- PR #1176: Use new RMM API based on Cython
- PR #1219: Adding custom bench_func and verbose logging to cuml.benchmark
- PR #1247: Improved MNMG RF error checking

## Bug Fixes

- PR #1231: RF respect number of cuda streams from cuml handle
- PR #1230: Rf bugfix memleak in regression
- PR #1208: compile dbscan bug
- PR #1016: Use correct libcumlprims version in GPU CI
- PR #1040: Update version of numba in development conda yaml files
- PR #1043: Updates to accomodate cuDF python code reorganization
- PR #1044: Remove nvidia driver installation from ci/cpu/build.sh
- PR #991: Barnes Hut TSNE Memory Issue Fixes
- PR #1075: Pinning Dask version for consistent CI results
- PR #990: Barnes Hut TSNE Memory Issue Fixes
- PR #1066: Using proper set of workers to destroy nccl comms
- PR #1072: Remove pip requirements and setup
- PR #1074: Fix flake8 CI style check
- PR #1087: Accuracy improvement for sqrt/log in RF max_feature
- PR #1088: Change straggling numba python allocations to use RMM
- PR #1106: Pinning Distributed version to match Dask for consistent CI results
- PR #1116: TSNE CUDA 10.1 Bug Fixes
- PR #1132: DBSCAN Batching Bug Fix
- PR #1162: DASK RF random seed bug fix
- PR #1164: Fix check_dtype arg handling for input_to_dev_array
- PR #1171: SVM prediction bug fix
- PR #1177: Update dask and distributed to 2.5
- PR #1204: Fix SVM crash on Turing
- PR #1199: Replaced sprintf() with snprintf() in THROW()
- PR #1205: Update dask-cuda in yml envs
- PR #1211: Fixing Dask k-means transform bug and adding test
- PR #1236: Improve fix for SMO solvers potential crash on Turing
- PR #1251: Disable compiler optimization for CUDA 10.1 for distance prims
- PR #1260: Small bugfix for major conversion in input_utils
- PR #1276: Fix float64 prediction crash in test_random_forest

# cuML 0.9.0 (21 Aug 2019)

## New Features

- PR #894: Convert RF to treelite format
- PR #826: Jones transformation of params for ARIMA models timeSeries ml-prim
- PR #697: Silhouette Score metric ml-prim
- PR #674: KL Divergence metric ml-prim
- PR #787: homogeneity, completeness and v-measure metrics ml-prim
- PR #711: Mutual Information metric ml-prim
- PR #724: Entropy metric ml-prim
- PR #766: Expose score method based on inertia for KMeans
- PR #823: prims: cluster dispersion metric
- PR #816: Added inverse_transform() for LabelEncoder
- PR #789: prims: sampling without replacement
- PR #813: prims: Col major istance prim
- PR #635: Random Forest & Decision Tree Regression (Single-GPU)
- PR #819: Forest Inferencing Library (FIL)
- PR #829: C++: enable nvtx ranges
- PR #835: Holt-Winters algorithm
- PR #837: treelite for decision forest exchange format
- PR #871: Wrapper for FIL
- PR #870: make_blobs python function
- PR #881: wrappers for accuracy_score and adjusted_rand_score functions
- PR #840: Dask RF classification and regression
- PR #870: make_blobs python function
- PR #879: import of treelite models to FIL
- PR #892: General Gram matrices prim
- PR #883: Adding MNMG Kmeans
- PR #930: Dask RF
- PR #882: TSNE - T-Distributed Stochastic Neighbourhood Embedding
- PR #624: Internals API & Graph Based Dimensionality Reductions Callback
- PR #926: Wrapper for FIL
- PR #994: Adding MPI comm impl for testing / benchmarking MNMG CUDA
- PR #960: Enable using libcumlprims for MG algorithms/prims

## Improvements
- PR #822: build: build.sh update to club all make targets together
- PR #807: Added development conda yml files
- PR #840: Require cmake >= 3.14
- PR #832: Stateless Decision Tree and Random Forest API
- PR #857: Small modifications to comms for utilizing IB w/ Dask
- PR #851: Random forest Stateless API wrappers
- PR #865: High Performance RF
- PR #895: Pretty prints arguments!
- PR #920: Add an empty marker kernel for tracing purposes
- PR #915: syncStream added to cumlCommunicator
- PR #922: Random Forest support in FIL
- PR #911: Update headers to credit CannyLabs BH TSNE implementation
- PR #918: Streamline CUDA_REL environment variable
- PR #924: kmeans: updated APIs to be stateless, refactored code for mnmg support
- PR #950: global_bias support in FIL
- PR #773: Significant improvements to input checking of all classes and common input API for Python
- PR #957: Adding docs to RF & KMeans MNMG. Small fixes for release
- PR #965: Making dask-ml a hard dependency
- PR #976: Update api.rst for new 0.9 classes
- PR #973: Use cudaDeviceGetAttribute instead of relying on cudaDeviceProp object being passed
- PR #978: Update README for 0.9
- PR #1009: Fix references to notebooks-contrib
- PR #1015: Ability to control the number of internal streams in cumlHandle_impl via cumlHandle
- PR #1175: Add more modules to docs ToC

## Bug Fixes

- PR #923: Fix misshapen level/trend/season HoltWinters output
- PR #831: Update conda package dependencies to cudf 0.9
- PR #772: Add missing cython headers to SGD and CD
- PR #849: PCA no attribute trans_input_ transform bug fix
- PR #869: Removing incorrect information from KNN Docs
- PR #885: libclang installation fix for GPUCI
- PR #896: Fix typo in comms build instructions
- PR #921: Fix build scripts using incorrect cudf version
- PR #928: TSNE Stability Adjustments
- PR #934: Cache cudaDeviceProp in cumlHandle for perf reasons
- PR #932: Change default param value for RF classifier
- PR #949: Fix dtype conversion tests for unsupported cudf dtypes
- PR #908: Fix local build generated file ownerships
- PR #983: Change RF max_depth default to 16
- PR #987: Change default values for knn
- PR #988: Switch to exact tsne
- PR #991: Cleanup python code in cuml.dask.cluster
- PR #996: ucx_initialized being properly set in CommsContext
- PR #1007: Throws a well defined error when mutigpu is not enabled
- PR #1018: Hint location of nccl in build.sh for CI
- PR #1022: Using random_state to make K-Means MNMG tests deterministic
- PR #1034: Fix typos and formatting issues in RF docs
- PR #1052: Fix the rows_sample dtype to float

# cuML 0.8.0 (27 June 2019)

## New Features

- PR #652: Adjusted Rand Index metric ml-prim
- PR #679: Class label manipulation ml-prim
- PR #636: Rand Index metric ml-prim
- PR #515: Added Random Projection feature
- PR #504: Contingency matrix ml-prim
- PR #644: Add train_test_split utility for cuDF dataframes
- PR #612: Allow Cuda Array Interface, Numba inputs and input code refactor
- PR #641: C: Separate C-wrapper library build to generate libcuml.so
- PR #631: Add nvcategory based ordinal label encoder
- PR #681: Add MBSGDClassifier and MBSGDRegressor classes around SGD
- PR #705: Quasi Newton solver and LogisticRegression Python classes
- PR #670: Add test skipping functionality to build.sh
- PR #678: Random Forest Python class
- PR #684: prims: make_blobs primitive
- PR #673: prims: reduce cols by key primitive
- PR #812: Add cuML Communications API & consolidate Dask cuML

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
- PR #675: atomicMin/Max(float, double) with integer atomics and bit flipping
- PR #677: build: 'deep-clean' to build.sh to clean faiss build as well
- PR #683: Use stateless c++ API in KNN so that it can be pickled properly
- PR #686: Use stateless c++ API in UMAP so that it can be pickled properly
- PR #695: prims: Refactor pairwise distance
- PR #707: Added stress test and updated documentation for RF
- PR #701: Added emacs temporary file patterns to .gitignore
- PR #606: C++: Added tests for host_buffer and improved device_buffer and host_buffer implementation
- PR #726: Updated RF docs and stress test
- PR #730: Update README and RF docs for 0.8
- PR #744: Random projections generating binomial on device. Fixing tests.
- PR #741: Update API docs for 0.8
- PR #754: Pickling of UMAP/KNN
- PR #753: Made PCA and TSVD picklable
- PR #746: LogisticRegression and QN API docstrings
- PR #820: Updating DEVELOPER GUIDE threading guidelines

## Bug Fixes
- PR #584: Added missing virtual destructor to deviceAllocator and hostAllocator
- PR #620: C++: Removed old unit-test files in ml-prims
- PR #627: C++: Fixed dbscan crash issue filed in 613
- PR #640: Remove setuptools from conda run dependency
- PR #646: Update link in contributing.md
- PR #649: Bug fix to LinAlg::reduce_rows_by_key prim filed in issue #648
- PR #666: fixes to gitutils.py to resolve both string decode and handling of uncommitted files
- PR #676: Fix template parameters in `bernoulli()` implementation.
- PR #685: Make CuPy optional to avoid nccl conda package conflicts
- PR #687: prims: updated tolerance for reduce_cols_by_key unit-tests
- PR #689: Removing extra prints from NearestNeighbors cython
- PR #718: Bug fix for DBSCAN and increasing batch size of sgd
- PR #719: Adding additional checks for dtype of the data
- PR #736: Bug fix for RF wrapper and .cu print function
- PR #547: Fixed issue if C++ compiler is specified via CXX during configure.
- PR #759: Configure Sphinx to render params correctly
- PR #762: Apply threshold to remove flakiness of UMAP tests.
- PR #768: Fixing memory bug from stateless refactor
- PR #782: Nearest neighbors checking properly whether memory should be freed
- PR #783: UMAP was using wrong size for knn computation
- PR #776: Hotfix for self.variables in RF
- PR #777: Fix numpy input bug
- PR #784: Fix jit of shuffle_idx python function
- PR #790: Fix rows_sample input type for RF
- PR #793: Fix for dtype conversion utility for numba arrays without cupy installed
- PR #806: Add a seed for sklearn model in RF test file
- PR #843: Rf quantile fix

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
- PR #721: Added pickle tests for DBSCAN and Random Projections

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
