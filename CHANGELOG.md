# cuML 0.19.0 (Date TBD)

Please see https://github.com/rapidsai/cuml/releases/tag/v0.19.0a for the latest changes to this development branch.

# cuML 0.18.0 (24 Feb 2021)

## Breaking Changes 🚨

- cuml.experimental SHAP improvements (#3433) @dantegd
- Enable feature sampling for the experimental backend of Random Forest (#3364) @vinaydes
- re-enable cuML&#39;s copyright checker script (#3363) @teju85
- Batched Silhouette Score (#3362) @divyegala
- Update failing MNMG tests (#3348) @viclafargue
- Rename print_summary() of Dask RF to get_summary_text(); it now returns string to the client (#3341) @hcho3
- Rename dump_as_json() -&gt; get_json(); expose it from Dask RF (#3340) @hcho3
- MNMG KNN consolidation (#3307) @viclafargue
- Return confusion matrix as int unless float weights are used (#3275) @lowener
- Approximate Nearest Neighbors (#2780) @viclafargue

## Bug Fixes 🐛

- HOTFIX Add ucx-proc package back that got lost during an auto merge conflict (#3551) @dantegd
- Non project-flash CI ml test 18.04 issue debugging and bugfixing (#3495) @dantegd
- Temporarily xfail KBinsDiscretizer uniform tests (#3494) @wphicks
- Fix illegal memory accesses when NITEMS &gt; 1, and nrows % NITEMS != 0. (#3480) @canonizer
- Update call to dask client persist (#3474) @dantegd
- Adding warning for IVFPQ (#3472) @viclafargue
- Fix failing sparse NN test in CI by allowing small number of index discrepancies (#3454) @cjnolet
- Exempting thirdparty code from copyright checks (#3453) @lowener
- Relaxing Batched SilhouetteScore Test Constraint (#3452) @divyegala
- Mark kbinsdiscretizer quantile tests as xfail (#3450) @wphicks
- Fixing documentation on SimpleImputer (#3447) @lowener
- Skipping IVFPQ (#3429) @viclafargue
- Adding tol to dask test_kmeans (#3426) @lowener
- Fix memory bug for SVM with large n_rows (#3420) @tfeher
- Allow linear regression for  with CUDA &gt;=11.0 (#3417) @wphicks
- Fix vectorizer tests by restoring sort behavior in groupby (#3416) @JohnZed
- Ensure make_classification respects output type (#3415) @wphicks
- Clean Up `#include` Dependencies (#3402) @mdemoret-nv
- Fix Nearest Neighbor Stress Test (#3401) @lowener
- Fix array_equal in tests (#3400) @viclafargue
- Improving Copyright Check When Not Running in CI (#3398) @mdemoret-nv
- Also xfail zlib errors when downloading newsgroups data (#3393) @JohnZed
- Fix for ANN memory release bug (#3391) @viclafargue
- XFail Holt Winters test where statsmodels has known issues with gcc 9.3.0 (#3385) @JohnZed
- FIX Update cupy to &gt;= 7.8 and remove unused build.sh script (#3378) @dantegd
- re-enable cuML&#39;s copyright checker script (#3363) @teju85
- Update failing MNMG tests (#3348) @viclafargue
- Rename print_summary() of Dask RF to get_summary_text(); it now returns string to the client (#3341) @hcho3
- Fixing `make_blobs` to Respect the Global Output Type (#3339) @mdemoret-nv
- Fix permutation explainer (#3332) @RAMitchell
- k-means bug fix in debug build (#3321) @akkamesh
- Fix for default arguments of PCA (#3320) @lowener
- Provide workaround for cupy.percentile bug (#3315) @wphicks
- Fix SVR unit test parameter (#3294) @tfeher
- Add xfail on fetching 20newsgroup dataset (test_naive_bayes) (#3291) @lowener
- Remove unused keyword in PorterStemmer code (#3289) @wphicks
- Remove static specifier in DecisionTree unit test for C++14 compliance (#3281) @wphicks
- Correct pure virtual declaration in manifold_inputs_t (#3279) @wphicks

## Documentation 📖

- Correct import path in docs for experimental preprocessing features (#3488) @wphicks
- Minor doc updates for 0.18 (#3475) @JohnZed
- Improve Python Docs with Default Role (#3445) @mdemoret-nv
- Fixing Python Documentation Errors and Warnings (#3428) @mdemoret-nv
- Remove outdated references to changelog in CONTRIBUTING.md (#3328) @wphicks
- Adding highlighting to bibtex in readme (#3296) @cjnolet

## New Features 🚀

- Improve runtime performance of RF to Treelite conversion (#3410) @wphicks
- Parallelize Treelite to FIL conversion over trees (#3396) @wphicks
- Parallelize RF to Treelite conversion over trees (#3395) @wphicks
- Allow saving Dask RandomForest models immediately after training (fixes #3331) (#3388) @jameslamb
- genetic programming initial structures (#3387) @teju85
- MNMG DBSCAN (#3382) @Nyrio
- FIL to use L1 cache when input columns don&#39;t fit into shared memory (#3370) @levsnv
- Enable feature sampling for the experimental backend of Random Forest (#3364) @vinaydes
- Batched Silhouette Score (#3362) @divyegala
- Rename dump_as_json() -&gt; get_json(); expose it from Dask RF (#3340) @hcho3
- Exposing model_selection in a similar way to scikit-learn (#3329) @ptartan21
- Promote IncrementalPCA from experimental in 0.18 release (#3327) @lowener
- Create labeler.yml (#3324) @jolorunyomi
- Add slow high-precision mode to KNN (#3304) @wphicks
- Sparse TSNE (#3293) @divyegala
- Sparse Generalized SPMV (semiring) Primitive (#3146) @cjnolet
- Multiclass meta estimator wrappers and multiclass SVC (#3092) @tfeher
- Approximate Nearest Neighbors (#2780) @viclafargue
- Add KNN parameter to t-SNE (#2592) @aleksficek

## Improvements 🛠️

- Update stale GHA with exemptions &amp; new labels (#3507) @mike-wendt
- Add GHA to mark issues/prs as stale/rotten (#3500) @Ethyling
- Fix naive bayes inputs (#3448) @cjnolet
- Prepare Changelog for Automation (#3442) @ajschmidt8
- cuml.experimental SHAP improvements (#3433) @dantegd
- Speed up knn tests (#3411) @JohnZed
- Replacing sklearn functions with cuml in RF MNMG notebook (#3408) @lowener
- Auto-label PRs based on their content (#3407) @jolorunyomi
- Use stable 1.0.0 version of Treelite (#3394) @hcho3
- API update to match RAFT PR #120 (#3386) @drobison00
- Update linear models to use RMM memory allocation (#3365) @lowener
- Updating dense pairwise distance enum names (#3352) @cjnolet
- Upgrade Treelite module (#3316) @hcho3
- Removed FIL node types with `_t` suffix (#3314) @canonizer
- MNMG KNN consolidation (#3307) @viclafargue
- Updating PyTests to Stay Below 4 Gb Limit (#3306) @mdemoret-nv
- Refactoring: move internal FIL interface to a separate file (#3292) @canonizer
- Return confusion matrix as int unless float weights are used (#3275) @lowener
- 018 add unfitted error pca &amp; tests on IPCA (#3272) @lowener
- Linear models predict function consolidation (#3256) @dantegd
- Preparing sparse primitives for movement to RAFT (#3157) @cjnolet

# cuML 0.17.0 (10 Dec 2020)

## New Features
- PR #3164: Expose silhouette score in Python
- PR #3160: Least Angle Regression (experimental)
- PR #2659: Add initial max inner product sparse knn
- PR #3092: Multiclass meta estimator wrappers and multiclass SVC
- PR #2836: Refactor UMAP to accept sparse inputs
- PR #2894: predict_proba in FIL C++ for XGBoost-style multi-class models
- PR #3126: Experimental versions of GPU accelerated Kernel and Permutation SHAP

## Improvements
- PR #3077: Improve runtime for test_kmeans
- PR #3070: Speed up dask/test_datasets tests
- PR #3075: Speed up test_linear_model tests
- PR #3078: Speed up test_incremental_pca tests
- PR #2902: `matrix/matrix.cuh` in RAFT namespacing
- PR #2903: Moving linalg's gemm, gemv, transpose to RAFT namespaces
- PR #2905: `stats` prims `mean_center`, `sum` to RAFT namespaces
- PR #2904: Moving `linalg` basic math ops to RAFT namespaces
- PR #2956: Follow cuML array conventions in ARIMA and remove redundancy
- PR #3000: Pin cmake policies to cmake 3.17 version, bump project version to 0.17
- PR #3083: Improving test_make_blobs testing time
- PR #3223: Increase default SVM kernel cache to 2000 MiB
- PR #2906: Moving `linalg` decomp to RAFT namespaces
- PR #2988: FIL: use tree-per-class reduction for GROVE_PER_CLASS_FEW_CLASSES
- PR #2996: Removing the max_depth restriction for switching to the batched backend
- PR #3004: Remove Single Process Multi GPU (SPMG) code
- PR #3032: FIL: Add optimization parameter `blocks_per_sm` that will help all but tiniest models
- PR #3044: Move leftover `linalg` and `stats` to RAFT namespaces
- PR #3067: Deleting prims moved to RAFT and updating header paths
- PR #3074: Reducing dask coordinate descent test runtime
- PR #3096: Avoid memory transfers in CSR WeakCC for DBSCAN
- PR #3088: More readable and robust FIL C++ test management
- PR #3052: Speeding up MNMG KNN Cl&Re testing
- PR #3115: Speeding up MNMG UMAP testing
- PR #3112: Speed test_array
- PR #3111: Adding Cython to Code Coverage
- PR #3129: Update notebooks README
- PR #3002: Update flake8 Config To With Per File Settings
- PR #3135: Add QuasiNewton tests
- PR #3040: Improved Array Conversion with CumlArrayDescriptor and Decorators
- PR #3134: Improving the Deprecation Message Formatting in Documentation
- PR #3154: Adding estimator pickling demo notebooks (and docs)
- PR #3151: MNMG Logistic Regression via dask-glm
- PR #3113: Add tags and prefered memory order tags to estimators
- PR #3137: Reorganize Pytest Config and Add Quick Run Option
- PR #3144: Adding Ability to Set Arbitrary Cmake Flags in ./build.sh
- PR #3155: Eliminate unnecessary warnings from random projection test
- PR #3176: Add probabilistic SVM tests with various input array types
- PR #3180: FIL: `blocks_per_sm` support in Python
- PR #3186: Add gain to RF JSON dump
- PR #3219: Update CI to use XGBoost 1.3.0 RCs
- PR #3221: Update contributing doc for label support
- PR #3177: Make Multinomial Naive Bayes inherit from `ClassifierMixin` and use it for score
- PR #3241: Updating RAFT to latest
- PR #3240: Minor doc updates
- PR #3275: Return confusion matrix as int unless float weights are used

## Bug Fixes
- PR #3218: Specify dependency branches in conda dev environment to avoid pip resolver issue
- PR #3196: Disable ascending=false path for sortColumnsPerRow
- PR #3051: MNMG KNN Cl&Re fix + multiple improvements
- PR #3179: Remove unused metrics.cu file
- PR #3069: Prevent conversion of DataFrames to Series in preprocessing
- PR #3065: Refactoring prims metrics function names from camelcase to underscore format
- PR #3033: Splitting ml metrics to individual files
- PR #3072: Fusing metrics and score directories in src_prims
- PR #3037: Avoid logging deadlock in multi-threaded C code
- PR #2983: Fix seeding of KISS99 RNG
- PR #3011: Fix unused initialize_embeddings parameter in Barnes-Hut t-SNE
- PR #3008: Check number of columns in check_array validator
- PR #3012: Increasing learning rate for SGD log loss and invscaling pytests
- PR #2950: Fix includes in UMAP
- PR #3194: Fix cuDF to cuPy conversion (missing value)
- PR #3021: Fix a hang in cuML RF experimental backend
- PR #3039: Update RF and decision tree parameter initializations in benchmark codes
- PR #3060: Speed up test suite `test_fil`
- PR #3061: Handle C++ exception thrown from FIL predict
- PR #3073: Update mathjax CDN URL for documentation
- PR #3062: Bumping xgboost version to match cuml version
- PR #3084: Fix artifacts in t-SNE results
- PR #3086: Reverting FIL Notebook Testing
- PR #3192: Enable pipeline usage for OneHotEncoder and LabelEncoder
- PR #3114: Fixed a typo in SVC's predict_proba AttributeError
- PR #3117: Fix two crashes in experimental RF backend
- PR #3119: Fix memset args for benchmark
- PR #3130: Return Python string from `dump_as_json()` of RF
- PR #3132: Add `min_samples_split` + Rename `min_rows_per_node` -> `min_samples_leaf`
- PR #3136: Fix stochastic gradient descent example
- PR #3152: Fix access to attributes of individual NB objects in dask NB
- PR #3156: Force local conda artifact install
- PR #3162: Removing accidentally checked in debug file
- PR #3191: Fix __repr__ function for preprocessing models
- PR #3175: Fix gtest pinned cmake version for build from source option
- PR #3182: Fix a bug in MSE metric calculation
- PR #3187: Update docstring to document behavior of `bootstrap=False`
- PR #3215: Add a missing `__syncthreads()`
- PR #3246: Fix MNMG KNN doc (adding batch_size)
- PR #3185: Add documentation for Distributed TFIDF Transformer
- PR #3190: Fix Attribute error on ICPA #3183 and PCA input type
- PR #3208: Fix EXITCODE override in notebook test script
- PR #3250: Fixing label binarizer bug with multiple partitions
- PR #3214: Correct flaky silhouette score test by setting atol
- PR #3216: Ignore splits that do not satisfy constraints
- PR #3239: Fix intermittent dask random forest failure
- PR #3243: Avoid unnecessary split for degenerate case where all labels are identical
- PR #3245: Rename `rows_sample` -> `max_samples` to be consistent with sklearn's RF
- PR #3282: Add secondary test to kernel explainer pytests for stability in Volta

# cuML 0.16.0 (23 Oct 2020)

## New Features
- PR #2922: Install RAFT headers with cuML
- PR #2909: Update allgatherv for compatibility with latest RAFT
- PR #2677: Ability to export RF trees as JSON
- PR #2698: Distributed TF-IDF transformer
- PR #2476: Porter Stemmer
- PR #2789: Dask LabelEncoder
- PR #2152: add FIL C++ benchmark
- PR #2638: Improve cython build with custom `build_ext`
- PR #2866: Support XGBoost-style multiclass models (gradient boosted decision trees) in FIL C++
- PR #2874: Issue warning for degraded accuracy with float64 models in Treelite
- PR #2881: Introduces experimental batched backend for random forest
- PR #2916: Add SKLearn multi-class GBDT model support in FIL

## Improvements
- PR #2947: Add more warnings for accuracy degradation with 64-bit models
- PR #2873: Remove empty marker kernel code for NVTX markers
- PR #2796: Remove tokens of length 1 by default for text vectorizers
- PR #2741: Use rapids build packages in conda environments
- PR #2735: Update seed to random_state in random forest and associated tests
- PR #2739: Use cusparse_wrappers.h from RAFT
- PR #2729: Replace `cupy.sparse` with `cupyx.scipy.sparse`
- PR #2749: Correct docs for python version used in cuml_dev conda environment
- PR #2747: Adopting raft::handle_t and raft::comms::comms_t in cuML
- PR #2762: Fix broken links and provide minor edits to docs
- PR #2723: Support and enable convert_dtype in estimator predict
- PR #2758: Match sklearn's default n_components behavior for PCA
- PR #2770: Fix doxygen version during cmake
- PR #2766: Update default RandomForestRegressor score function to use r2
- PR #2775: Enablinbg mg gtests w/ raft mpi comms
- PR #2783: Add pytest that will fail when GPU IDs in Dask cluster are not unique
- PR #2784: Add SparseCumlArray container for sparse index/data arrays
- PR #2785: Add in cuML-specific dev conda dependencies
- PR #2778: Add README for FIL
- PR #2799: Reenable lightgbm test with lower (1%) proba accuracy
- PR #2800: Align cuML's spdlog version with RMM's
- PR #2824: Make data conversions warnings be debug level
- PR #2835: Rng prims, utils, and dependencies in RAFT
- PR #2541: Improve Documentation Examples and Source Linking
- PR #2837: Make the FIL node reorder loop more obvious
- PR #2849: make num_classes significant in FLOAT_SCALAR case
- PR #2792: Project flash (new build process) script changes
- PR #2850: Clean up unused params in paramsPCA
- PR #2871: Add timing function to utils
- PR #2863: in FIL, rename leaf_value_t enums to more descriptive
- PR #2867: improve stability of FIL benchmark measurements
- PR #2798: Add python tests for FIL multiclass classification of lightgbm models
- PR #2892: Update ci/local/README.md
- PR #2910: Adding Support for CuPy 8.x
- PR #2914: Add tests for XGBoost multi-class models in FIL
- PR #2622: Simplify tSNE perplexity search
- PR #2930: Pin libfaiss to <=1.6.3
- PR #2928: Updating Estimators Derived from Base for Consistency
- PR #2942: Adding `cuml.experimental` to the Docs
- PR #3010: Improve gpuCI Scripts
- PR #3141: Move DistanceType enum to RAFT

## Bug Fixes
- PR #2973: Allow data imputation for nan values
- PR #2982: Adjust kneighbors classifier test threshold to avoid intermittent failure
- PR #2885: Changing test target for NVTX wrapper test
- PR #2882: Allow import on machines without GPUs
- PR #2875: Bug fix to enable colorful NVTX markers
- PR #2744: Supporting larger number of classes in KNeighborsClassifier
- PR #2769: Remove outdated doxygen options for 1.8.20
- PR #2787: Skip lightgbm test for version 3 and above temporarily
- PR #2805: Retain index in stratified splitting for dataframes
- PR #2781: Use Python print to correctly redirect spdlogs when sys.stdout is changed
- PR #2787: Skip lightgbm test for version 3 and above temporarily
- PR #2813: Fix memory access in generation of non-row-major random blobs
- PR #2810: Update Rf MNMG threshold to prevent sporadic test failure
- PR #2808: Relax Doxygen version required in CMake to coincide with integration repo
- PR #2818: Fix parsing of singlegpu option in build command
- PR #2827: Force use of whole dataset when sample bootstrapping is disabled
- PR #2829: Fixing description for labels in docs and removing row number constraint from PCA xform/inverse_xform
- PR #2832: Updating stress tests that fail with OOM
- PR #2831: Removing repeated capture and parameter in lambda function
- PR #2847: Workaround for TSNE lockup, change caching preference.
- PR #2842: KNN index preprocessors were using incorrect n_samples
- PR #2848: Fix typo in Python docstring for UMAP
- PR #2856: Fix LabelEncoder for filtered input
- PR #2855: Updates for RMM being header only
- PR #2844: Fix for OPG KNN Classifier & Regressor
- PR #2880: Fix bugs in Auto-ARIMA when s==None
- PR #2877: TSNE exception for n_components > 2
- PR #2879: Update unit test for LabelEncoder on filtered input
- PR #2932: Marking KBinsDiscretizer pytests as xfail
- PR #2925: Fixing Owner Bug When Slicing CumlArray Objects
- PR #2931: Fix notebook error handling in gpuCI
- PR #2941: Fixing dask tsvd stress test failure
- PR #2943: Remove unused shuffle_features parameter
- PR #2940: Correcting labels meta dtype for `cuml.dask.make_classification`
- PR #2965: Notebooks update
- PR #2955: Fix for conftest for singlegpu build
- PR #2968: Remove shuffle_features from RF param names
- PR #2957: Fix ols test size for stability
- PR #2972: Upgrade Treelite to 0.93
- PR #2981: Prevent unguarded import of sklearn in SVC
- PR #2984: Fix GPU test scripts gcov error
- PR #2990: Reduce MNMG kneighbors regressor test threshold
- PR #2997: Changing ARIMA `get/set_params` to `get/set_fit_params`

# cuML 0.15.0 (26 Aug 2020)

## New Features
- PR #2581: Added model persistence via joblib in each section of estimator_intro.ipynb
- PR #2554: Hashing Vectorizer and general vectorizer improvements
- PR #2240: Making Dask models pickleable
- PR #2267: CountVectorizer estimator
- PR #2261: Exposing new FAISS metrics through Python API
- PR #2287: Single-GPU TfidfTransformer implementation
- PR #2289: QR SVD solver for MNMG PCA
- PR #2312: column-major support for make_blobs
- PR #2172: Initial support for auto-ARIMA
- PR #2394: Adding cosine & correlation distance for KNN
- PR #2392: PCA can accept sparse inputs, and sparse prim for computing covariance
- PR #2465: Support pandas 1.0+
- PR #2550: Single GPU Target Encoder
- PR #2519: Precision recall curve using cupy
- PR #2500: Replace UMAP functionality dependency on nvgraph with RAFT Spectral Clustering
- PR #2502: cuML Implementation of `sklearn.metrics.pairwise_distances`
- PR #2520: TfidfVectorizer estimator
- PR #2211: MNMG KNN Classifier & Regressor
- PR #2461: Add KNN Sparse Output Functionality
- PR #2615: Incremental PCA
- PR #2594: Confidence intervals for ARIMA forecasts
- PR #2607: Add support for probability estimates in SVC
- PR #2618: SVM class and sample weights
- PR #2635: Decorator to generate docstrings with autodetection of parameters
- PR #2270: Multi class MNMG RF
- PR #2661: CUDA-11 support for single-gpu code
- PR #2322: Sparse FIL forests with 8-byte nodes
- PR #2675: Update conda recipes to support CUDA 11
- PR #2645: Add experimental, sklearn-based preprocessing

## Improvements
- PR #2336: Eliminate `rmm.device_array` usage
- PR #2262: Using fully shared PartDescriptor in MNMG decomposiition, linear models, and solvers
- PR #2310: Pinning ucx-py to 0.14 to make 0.15 CI pass
- PR #1945: enable clang tidy
- PR #2339: umap performance improvements
- PR #2308: Using fixture for Dask client to eliminate possiblity of not closing
- PR #2345: make C++ logger level definition to be the same as python layer
- PR #2329: Add short commit hash to conda package name
- PR #2362: Implement binary/multi-classification log loss with cupy
- PR #2363: Update threshold and make other changes for stress tests
- PR #2371: Updating MBSGD tests to use larger batches
- PR #2380: Pinning libcumlprims version to ease future updates
- PR #2405: Remove references to deprecated RMM headers.
- PR #2340: Import ARIMA in the root init file and fix the `test_fit_function` test
- PR #2408: Install meta packages for dependencies
- PR #2417: Move doc customization scripts to Jenkins
- PR #2427: Moving MNMG decomposition to cuml
- PR #2433: Add libcumlprims_mg to CMake
- PR #2420: Add and set convert_dtype default to True in estimator fit methods
- PR #2411: Refactor Mixin classes and use in classifier/regressor estimators
- PR #2442: fix setting RAFT_DIR from the RAFT_PATH env var
- PR #2469: Updating KNN c-api to document all arguments
- PR #2453: Add CumlArray to API doc
- PR #2440: Use Treelite Conda package
- PR #2403: Support for input and output type consistency in logistic regression predict_proba
- PR #2473: Add metrics.roc_auc_score to API docs. Additional readability and minor docs bug fixes
- PR #2468: Add `_n_features_in_` attribute to all single GPU estimators that implement fit
- PR #2489: Removing explicit FAISS build and adding dependency on libfaiss conda package
- PR #2480: Moving MNMG glm and solvers to cuml
- PR #2490: Moving MNMG KMeans to cuml
- PR #2483: Moving MNMG KNN to cuml
- PR #2492: Adding additional assertions to mnmg nearest neighbors pytests
- PR #2439: Update dask RF code to have print_detailed function
- PR #2431: Match output of classifier predict with target dtype
- PR #2237: Refactor RF cython code
- PR #2513: Fixing LGTM Analysis Issues
- PR #2099: Raise an error when float64 data is used with dask RF
- PR #2522: Renaming a few arguments in KNeighbors* to be more readable
- PR #2499: Provide access to `cuml.DBSCAN` core samples
- PR #2526: Removing PCA TSQR as a solver due to scalability issues
- PR #2536: Update conda upload versions for new supported CUDA/Python
- PR #2538: Remove Protobuf dependency
- PR #2553: Test pickle protocol 5 support
- PR #2570: Accepting single df or array input in train_test_split
- PR #2566: Remove deprecated cuDF from_gpu_matrix calls
- PR #2583: findpackage.cmake.in template for cmake dependencies
- PR #2577: Fully removing NVGraph dependency for CUDA 11 compatibility
- PR #2575: Speed up TfidfTransformer
- PR #2584: Removing dependency on sklearn's NotFittedError
- PR #2591: Generate benchmark datsets using `cuml.datasets`
- PR #2548: Fix limitation on number of rows usable with tSNE and refactor memory allocation
- PR #2589: including cuda-11 build fixes into raft
- PR #2599: Add Stratified train_test_split
- PR #2487: Set classes_ attribute during classifier fit
- PR #2605: Reduce memory usage in tSNE
- PR #2611: Adding building doxygen docs to gpu ci
- PR #2631: Enabling use of gtest conda package for build
- PR #2623: Fixing kmeans score() API to be compatible with Scikit-learn
- PR #2629: Add naive_bayes api docs
- PR #2643: 'dense' and 'sparse' values of `storage_type` for FIL
- PR #2691: Generic Base class attribute setter
- PR #2666: Update MBSGD documentation to mention that the model is experimental
- PR #2687: Update xgboost version to 1.2.0dev.rapidsai0.15
- PR #2684: CUDA 11 conda development environment yml and faiss patch
- PR #2648: Replace CNMeM with `rmm::mr::pool_memory_resource`.
- PR #2686: Improve SVM tests
- PR #2692: Changin LBFGS log level
- PR #2705: Add sum operator and base operator overloader functions to cumlarray
- PR #2701: Updating README + Adding ref to UMAP paper
- PR #2721: Update API docs
- PR #2730: Unpin cumlprims in conda recipes for release

## Bug Fixes
- PR #2369: Update RF code to fix set_params memory leak
- PR #2364: Fix for random projection
- PR #2373: Use Treelite Pip package in GPU testing
- PR #2376: Update documentation Links
- PR #2407: fixed batch count in DBScan for integer overflow case
- PR #2413: CumlArray and related methods updates to account for cuDF.Buffer contiguity update
- PR #2424: --singlegpu flag fix on build.sh script
- PR #2432: Using correct algo_name for UMAP in benchmark tests
- PR #2445: Restore access to coef_ property of Lasso
- PR #2441: Change p2p_enabled definition to work without ucx
- PR #2447: Drop `nvstrings`
- PR #2450: Update local build to use new gpuCI image
- PR #2454: Mark RF memleak test as XFAIL, because we can't detect memleak reliably
- PR #2455: Use correct field to store data type in `LabelEncoder.fit_transform`
- PR #2475: Fix typo in build.sh
- PR #2496: Fixing indentation for simulate_data in test_fil.py
- PR #2494: Set QN regularization strength consistent with scikit-learn
- PR #2486: Fix cupy input to kmeans init
- PR #2497: Changes to accomodate cuDF unsigned categorical changes
- PR #2209: Fix FIL benchmark for gpuarray-c input
- PR #2507: Import `treelite.sklearn`
- PR #2521: Fixing invalid smem calculation in KNeighborsCLassifier
- PR #2515: Increase tolerance for LogisticRegression test
- PR #2532: Updating doxygen in new MG headers
- PR #2521: Fixing invalid smem calculation in KNeighborsCLassifier
- PR #2515: Increase tolerance for LogisticRegression test
- PR #2545: Fix documentation of n_iter_without_progress in tSNE Python bindings
- PR #2543: Improve numerical stability of QN solver
- PR #2544: Fix Barnes-Hut tSNE not using specified post_learning_rate
- PR #2558: Disabled a long-running FIL test
- PR #2540: Update default value for n_epochs in UMAP to match documentation & sklearn API
- PR #2535: Fix issue with incorrect docker image being used in local build script
- PR #2542: Fix small memory leak in TSNE
- PR #2552: Fixed the length argument of updateDevice calls in RF test
- PR #2565: Fix cell allocation code to avoid loops in quad-tree. Prevent NaNs causing infinite descent
- PR #2563: Update scipy call for arima gradient test
- PR #2569: Fix for cuDF update
- PR #2508: Use keyword parameters in sklearn.datasets.make_* functions
- PR #2587: Attributes for estimators relying on solvers
- PR #2586: Fix SVC decision function data type
- PR #2573: Considering managed memory as device type on checking for KMeans
- PR #2574: Fixing include path in `tsvd_mg.pyx`
- PR #2506: Fix usage of CumlArray attributes on `cuml.common.base.Base`
- PR #2593: Fix inconsistency in train_test_split
- PR #2609: Fix small doxygen issues
- PR #2610: Remove cuDF tolist call
- PR #2613: Removing thresholds from kmeans score tests (SG+MG)
- PR #2616: Small test code fix for pandas dtype tests
- PR #2617: Fix floating point precision error in tSNE
- PR #2625: Update Estimator notebook to resolve errors
- PR #2634: singlegpu build option fixes
- PR #2641: [Breaking] Make `max_depth` in RF compatible with scikit-learn
- PR #2650: Make max_depth behave consistently for max_depth > 14
- PR #2651: AutoARIMA Python bug fix
- PR #2654: Fix for vectorizer concatenations
- PR #2655: Fix C++ RF predict function access of rows/samples array
- PR #2649: Cleanup sphinx doc warnings for 0.15
- PR #2668: Order conversion improvements to account for cupy behavior changes
- PR #2669: Revert PR 2655 Revert "Fixes C++ RF predict function"
- PR #2683: Fix incorrect "Bad CumlArray Use" error messages on test failures
- PR #2695: Fix debug build issue due to incorrect host/device method setup
- PR #2709: Fixing OneHotEncoder Overflow Error
- PR #2710: Fix SVC doc statement about predic_proba
- PR #2726: Return correct output type in QN
- PR #2711: Fix Dask RF failure intermittently
- PR #2718: Fix temp directory for py.test
- PR #2719: Set KNeighborsRegressor output dtype according to training target dtype
- PR #2720: Updates to outdated links
- PR #2722: Getting cuML covariance test passing w/ Cupy 7.8 & CUDA 11

# cuML 0.14.0 (03 Jun 2020)

## New Features
- PR #1994: Support for distributed OneHotEncoder
- PR #1892: One hot encoder implementation with cupy
- PR #1655: Adds python bindings for homogeneity score
- PR #1704: Adds python bindings for completeness score
- PR #1687: Adds python bindings for mutual info score
- PR #1980: prim: added a new write-only unary op prim
- PR #1867: C++: add logging interface support in cuML based spdlog
- PR #1902: Multi class inference in FIL C++ and importing multi-class forests from treelite
- PR #1906: UMAP MNMG
- PR #2067: python: wrap logging interface in cython
- PR #2083: Added dtype, order, and use_full_low_rank to MNMG `make_regression`
- PR #2074: SG and MNMG `make_classification`
- PR #2127: Added order to SG `make_blobs`, and switch from C++ to cupy based implementation
- PR #2057: Weighted k-means
- PR #2256: Add a `make_arima` generator
- PR #2245: ElasticNet, Lasso and Coordinate Descent MNMG
- PR #2242: Pandas input support with output as NumPy arrays by default
- PR #2551: Add cuML RF multiclass prediction using FIL from python
- PR #1728: Added notebook testing to gpuCI gpu build

## Improvements
- PR #1931: C++: enabled doxygen docs for all of the C++ codebase
- PR #1944: Support for dask_cudf.core.Series in _extract_partitions
- PR #1947: Cleaning up cmake
- PR #1927: Use Cython's `new_build_ext` (if available)
- PR #1946: Removed zlib dependency from cmake
- PR #1988: C++: cpp bench refactor
- PR #1873: Remove usage of nvstring and nvcat from LabelEncoder
- PR #1968: Update SVC SVR with cuML Array
- PR #1972: updates to our flow to use conda-forge's clang and clang-tools packages
- PR #1974: Reduce ARIMA testing time
- PR #1984: Enable Ninja build
- PR #1985: C++ UMAP parametrizable tests
- PR #2005: Adding missing algorithms to cuml benchmarks and notebook
- PR #2016: Add capability to setup.py and build.sh to fully clean all cython build files and artifacts
- PR #2044: A cuda-memcheck helper wrapper for devs
- PR #2018: Using `cuml.dask.part_utils.extract_partitions` and removing similar, duplicated code
- PR #2019: Enable doxygen build in our nightly doc build CI script
- PR #1996: Cythonize in parallel
- PR #2032: Reduce number of tests for MBSGD to improve CI running time
- PR #2031: Encapsulating UCX-py interactions in singleton
- PR #2029: Add C++ ARIMA log-likelihood benchmark
- PR #2085: Convert TSNE to use CumlArray
- PR #2051: Reduce the time required to run dask pca and dask tsvd tests
- PR #1981: Using CumlArray in kNN and DistributedDataHandler in dask kNN
- PR #2053: Introduce verbosity level in C++ layer instead of boolean `verbose` flag
- PR #2047: Make internal streams non-blocking w.r.t. NULL stream
- PR #2048: Random forest testing speedup
- PR #2058: Use CumlArray in Random Projection
- PR #2068: Updating knn class probabilities to use make_monotonic instead of binary search
- PR #2062: Adding random state to UMAP mnmg tests
- PR #2064: Speed-up K-Means test
- PR #2015: Renaming .h to .cuh in solver, dbscan and svm
- PR #2080: Improved import of sparse FIL forests from treelite
- PR #2090: Upgrade C++ build to C++14 standard
- PR #2089: CI: enabled cuda-memcheck on ml-prims unit-tests during nightly build
- PR #2128: Update Dask RF code to reduce the time required for GPU predict to run
- PR #2125: Build infrastructure to use RAFT
- PR #2131: Update Dask RF fit to use DistributedDataHandler
- PR #2055: Update the metrics notebook to use important cuML models
- PR #2095: Improved import of src_prims/utils.h, making it less ambiguous
- PR #2118: Updating SGD & mini-batch estimators to use CumlArray
- PR #2120: Speeding up dask RandomForest tests
- PR #1883: Use CumlArray in ARIMA
- PR #877: Adding definition of done criteria to wiki
- PR #2135: A few optimizations to UMAP fuzzy simplicial set
- PR #1914: Change the meaning of ARIMA's intercept to match the literature
- PR #2098: Renaming .h to .cuh in decision_tree, glm, pca
- PR #2150: Remove deprecated RMM calls in RMM allocator adapter
- PR #2146: Remove deprecated kalman filter
- PR #2151: Add pytest duration and pytest timeout
- PR #2156: Add Docker 19 support to local gpuci build
- PR #2178: Reduce duplicated code in RF
- PR #2124: Expand tutorial docs and sample notebook
- PR #2175: Allow CPU-only and dataset params for benchmark sweeps
- PR #2186: Refactor cython code to build OPG structs in common utils file
- PR #2180: Add fully single GPU singlegpu python build
- PR #2187: CMake improvements to manage conda environment dependencies
- PR #2185: Add has_sklearn function and use it in datasets/classification.
- PR #2193: Order-independent local shuffle in `cuml.dask.make_regression`
- PR #2204: Update python layer to use the logger interface
- PR #2184: Refoctor headers for holtwinters, rproj, tsvd, tsne, umap
- PR #2199: Remove unncessary notebooks
- PR #2195: Separating fit and transform calls in SG, MNMG PCA to save transform array memory consumption
- PR #2201: Re-enabling UMAP repro tests
- PR #2132: Add SVM C++ benchmarks
- PR #2196: Updates to benchmarks. Moving notebook
- PR #2208: Coordinate Descent, Lasso and ElasticNet CumlArray updates
- PR #2210: Updating KNN tests to evaluate multiple index partitions
- PR #2205: Use timeout to add 2 hour hard limit to dask tests
- PR #2212: Improve DBScan batch count / memory estimation
- PR #2213: Standardized include statements across all cpp source files, updated copyright on all modified files
- PR #2214: Remove utils folder and refactor to common folder
- PR #2220: Final refactoring of all src_prims header files following rules as specified in #1675
- PR #2225: input_to_cuml_array keep order option, test updates and cleanup
- PR #2244: Re-enable slow ARIMA tests as stress tests
- PR #2231: Using OPG structs from `cuml.common` in decomposition algorithms
- PR #2257: Update QN and LogisticRegression to use CumlArray
- PR #2259: Add CumlArray support to Naive Bayes
- PR #2252: Add benchmark for the Gram matrix prims
- PR #2263: Faster serialization for Treelite objects with RF
- PR #2264: Reduce build time for cuML by using make_blobs from libcuml++ interface
- PR #2269: Add docs targets to build.sh and fix python cuml.common docs
- PR #2271: Clarify doc for `_unique` default implementation in OneHotEncoder
- PR #2272: Add docs build.sh script to repository
- PR #2276: Ensure `CumlArray` provided `dtype` conforms
- PR #2281: Rely on cuDF's `Serializable` in `CumlArray`
- PR #2284: Reduce dataset size in SG RF notebook to reduce run time of sklearn
- PR #2285: Increase the threshold for elastic_net test in dask/test_coordinate_descent
- PR #2314: Update FIL default values, documentation and test
- PR #2316: 0.14 release docs additions and fixes
- PR #2320: Add prediction notes to RF docs
- PR #2323: Change verbose levels and parameter name to match Scikit-learn API
- PR #2324: Raise an error if n_bins > number of training samples in RF
- PR #2335: Throw a warning if treelite cannot be imported and `load_from_sklearn` is used

## Bug Fixes
- PR #1939: Fix syntax error in cuml.common.array
- PR #1941: Remove c++ cuda flag that was getting duplicated in CMake
- PR #1971: python: Correctly honor --singlegpu option and CUML_BUILD_PATH env variable
- PR #1969: Update libcumlprims to 0.14
- PR #1973: Add missing mg files for setup.py --singlegpu flag
- PR #1993: Set `umap_transform_reproducibility` tests to xfail
- PR #2004: Refactoring the arguments to `plant()` call
- PR #2017: Fixing memory issue in weak cc prim
- PR #2028: Skipping UMAP knn reproducibility tests until we figure out why its failing in CUDA 10.2
- PR #2024: Fixed cuda-memcheck errors with sample-without-replacement prim
- PR #1540: prims: support for custom math-type used for computation inside adjusted rand index prim
- PR #2077: dask-make blobs arguments to match sklearn
- PR #2059: Make all Scipy imports conditional
- PR #2078: Ignore negative cache indices in get_vecs
- PR #2084: Fixed cuda-memcheck errors with COO unit-tests
- PR #2087: Fixed cuda-memcheck errors with dispersion prim
- PR #2096: Fixed syntax error with nightly build command for memcheck unit-tests
- PR #2115: Fixed contingency matrix prim unit-tests for computing correct golden values
- PR #2107: Fix PCA transform
- PR #2109: input_to_cuml_array __cuda_array_interface__ bugfix
- PR #2117: cuDF __array__ exception small fixes
- PR #2139: CumlArray for adjusted_rand_score
- PR #2140: Returning self in fit model functions
- PR #2144: Remove GPU arch < 60 from CMake build
- PR #2153: Added missing namespaces to some Decision Tree files
- PR #2155: C++: fix doxygen build break
- PR #2161: Replacing depreciated bruteForceKnn
- PR #2162: Use stream in transpose prim
- PR #2165: Fit function test correction
- PR #2166: Fix handling of temp file in RF pickling
- PR #2176: C++: fix for adjusted rand index when input array is all zeros
- PR #2179: Fix clang tools version in libcuml recipe
- PR #2183: Fix RAFT in nightly package
- PR #2191: Fix placement of SVM parameter documentation and add examples
- PR #2212: Fix DBScan results (no propagation of labels through border points)
- PR #2215: Fix the printing of forest object
- PR #2217: Fix opg_utils naming to fix singlegpu build
- PR #2223: Fix bug in ARIMA C++ benchmark
- PR #2224: Temporary fix for CI until new Dask version is released
- PR #2228: Update to use __reduce_ex__ in CumlArray to override cudf.Buffer
- PR #2249: Fix bug in UMAP continuous target metrics
- PR #2258: Fix doxygen build break
- PR #2255: Set random_state for train_test_split function in dask RF
- PR #2275: Fix RF fit memory leak
- PR #2274: Fix parameter name verbose to verbosity in mnmg OneHotEncoder
- PR #2277: Updated cub repo path and branch name
- PR #2282: Fix memory leak in Dask RF concatenation
- PR #2301: Scaling KNN dask tests sample size with n GPUs
- PR #2293: Contiguity fixes for input_to_cuml_array and train_test_split
- PR #2295: Fix convert_to_dtype copy even with same dtype
- PR #2305: Fixed race condition in DBScan
- PR #2354: Fix broken links in README
- PR #2619: Explicitly skip raft test folder for pytest 6.0.0
- PR #2788: Set the minimum number of columns that can be sampled to 1 to fix 0 mem allocation error

# cuML 0.13.0 (31 Mar 2020)

## New Features
- PR #1777: Python bindings for entropy
- PR #1742: Mean squared error implementation with cupy
- PR #1817: Confusion matrix implementation with cupy (SNSG and MNMG)
- PR #1766: Mean absolute error implementation with cupy
- PR #1766: Mean squared log error implementation with cupy
- PR #1635: cuML Array shim and configurable output added to cluster methods
- PR #1586: Seasonal ARIMA
- PR #1683: cuml.dask make_regression
- PR #1689: Add framework for cuML Dask serializers
- PR #1709: Add `decision_function()` and `predict_proba()` for LogisticRegression
- PR #1714: Add `print_env.sh` file to gather important environment details
- PR #1750: LinearRegression CumlArray for configurable output
- PR #1814: ROC AUC score implementation with cupy
- PR #1767: Single GPU decomposition models configurable output
- PR #1646: Using FIL to predict in MNMG RF
- PR #1778: Make cuML Handle picklable
- PR #1738: cuml.dask refactor beginning and dask array input option for OLS, Ridge and KMeans
- PR #1874: Add predict_proba function to RF classifier
- PR #1815: Adding KNN parameter to UMAP
- PR #1978: Adding `predict_proba` function to dask RF

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
- PR #1873: Remove usage of nvstring and nvcat from LabelEncoder
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
- PR #1936: Update DaskRF regression test to xfail
- PR #1932: Isolating cause of make_blobs failure
- PR #1951: Dask Random forest regression CPU predict bug fix
- PR #1948: Adjust BatchedMargin margin and disable tests temporarily
- PR #1950: Fix UMAP test failure


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
