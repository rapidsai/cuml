cuml.accel: Zero Code Change Acceleration for Scikit-Learn, UMAP and HDBSCAN
============================================================================

Starting in RAPIDS 25.02.01, cuML offers a new way to accelerate existing code
based on Scikit-Learn, UMAP-Learn, and HDBSCAN. Instead of rewriting that code
to import equivalent cuML functionality, simply invoke your existing,
unaltered Python script as follows, and cuML will accelerate as much of the
code as possible with NVIDIA GPUs, falling back to CPU where necessary:

.. code-block:: console

    python -m cuml.accel unchanged_script.py

The same functionality is available in Jupyter notebooks using the
following magic at the beginning of the notebook (before other imports):

.. code-block::

   %load_ext cuml.accel
   import sklearn

You can see an example of this in
`KMeans Digits Notebook <zero_code_change_examples/plot_kmeans_digits.ipynb>`_, where an unmodified
Scikit-Learn example notebook is used to demonstrate how ``cuml.accel`` can be
used in Jupyter.

In any Python environment, the following code snippet can also be used to
activate ``cuml.accel`` if it is run prior to importing the module you wish to
accelerate:

.. code-block:: python

   from cuml.accel import install
   install()
   import sklearn

**``cuml.accel`` is currently a beta feature and will continue to improve over
time.**

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   zero-code-change-limitations.rst
   zero-code-change-benchmarks.rst
   zero_code_change_examples/plot_kmeans_digits.ipynb


FAQs
----

1. Why use cuml.accel instead of using cuML directly?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Many software lifecycles involve running code on a variety of hardware. Maybe
the data scientists developing a pipeline do not have access to NVIDIA GPUs,
but you want the cost and time savings of running that pipeline on NVIDIA GPUs
in production. Rather than going through a manual migration to cuML every time
the pipeline is updated, ``cuml.accel`` allows you to immediately deploy
unaltered Scikit-Learn, UMAP-Learn, and HDBSCAN code on NVIDIA GPUs.
Furthermore, ``cuml.accel`` will automatically fall back to CPU execution for
anything which is implemented in Scikit-Learn but not yet accelerated by cuML.

Additionally, ``cuml.accel`` offers a quick way to evaluate the minimum
acceleration cuML can provide for your workload without touching a line of
code.

2. Why use cuML directly instead of cuml.accel?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In many cases, ``cuml.accel`` offers enough of a performance boost on its own
that there is no need to migrate code to cuML. However, cuML's API offers a
variety of additional parameters that let you fine-tune GPU execution in order
to get the maximum possible performance out of NVIDIA GPUs. So for software
that will always be run with NVIDIA GPUs available, it may be worthwhile to
write your code directly with cuML.

Additionally, running code directly with cuML offers finer control over GPU
memory usage. ``cuml.accel`` will automatically use `unified or managed memory <https://developer.nvidia.com/blog/unified-memory-cuda-beginners/>`_
for allocations in order to reduce the risk of CUDA OOM errors. In
contrast, cuML defaults to ordinary device memory, which can offer improved
performance but requires slightly more care to avoid exhausting the GPU VRAM.
If you experience unexpectedly slow performance with ``cuml.accel``, you can
try disabling the use of unified memory with the ``--disable-uvm`` flag.

3. What does ``cuml.accel`` accelerate?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``cuml.accel`` is designed to provide zero code change acceleration of any
Scikit-Learn-like estimator which has an equivalent cuML implementation,
including estimators from Scikit-Learn, UMAP-Learn, and HDBSCAN. Currently,
the following estimators are mostly or entirely accelerated when run with
``cuml.accel``:

* UMAP-Learn
    * ``umap.UMAP``
* HDBSCAN
    * ``hdbscan.HDBSCAN``
* Scikit-Learn
    * ``sklearn.cluster.KMeans``
    * ``sklearn.cluster.DBSCAN``
    * ``sklearn.decomposition.PCA``
    * ``sklearn.decomposition.TruncatedSVD``
    * ``sklearn.ensemble.RandomForestClassifier``
    * ``sklearn.ensemble.RandomForestRegressor``
    * ``sklearn.kernel_ridge.KernelRidge``
    * ``sklearn.linear_model.LinearRegression``
    * ``sklearn.linear_model.LogisticRegression``
    * ``sklearn.linear_model.ElasticNet``
    * ``sklearn.linear_model.Ridge``
    * ``sklearn.linear_model.Lasso``
    * ``sklearn.manifold.TSNE``
    * ``sklearn.neighbors.NearestNeighbors``
    * ``sklearn.neighbors.KNeighborsClassifier``
    * ``sklearn.neighbors.KNeighborsRegressor``

This list will continue to expand as ``cuml.accel`` development
continues to cover all algorithms present in cuML.
Please see `Zero Code Change Limitations <zero-code-change-limitations.rst>`_
for known limitations.

If there are specific models that you are particularly interested in seeing
prioritized for cuML, please request them via an `issue <https://github.com/rapidsai/cuml/issues/new?template=feature_request.md>`_ in
the cuML Githb repository.

4. Will I get the same results as I do without ``cuml.accel``?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``cuml.accel`` is designed to provide *equivalent* results to the estimators
it acelerates, but the output may have small numerical differences. To be more
specific, measures of the quality of the results (accuracy,
trustworthiness, etc.) should be approximately as good or better than those
obtained without ``cuml.accel``, even if the exact output varies.

A baseline limitation for obtaining exact numerical equality is that in
highly parallel execution environments (e.g. GPUs), there is no guarantee that
floating point operations will happen in exactly the same order as in
non-parallel environments. This means that floating point arithmetic error
may propagate differently and lead to different outcomes. This can be
exacerbated by discretization operations in which values end up in
different categories based on floating point values.

Secondarily, some algorithms are implemented in a fundamentally different
way on GPU than on CPU in order to make efficient use of the GPU's highly
parallel compute capabilities. In such cases, ``cuml.accel`` will translate
hyperparameters appropriately to maintain equivalence with the CPU
implementation. Differences of this kind are noted in the corresponding entry
of `Zero Code Change Limitations <zero-code-change-limitations.rst>`_ for that
estimator.

If you discover a use case where the quality of results obtained with
``cuml.accel`` is worse than that obtained without, please `report it as a bug
<https://github.com/rapidsai/cuml/issues/new?template=bug_report.md>`_, and the
RAPIDS team will investigate.

5. How much faster is ``cuml.accel``?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This depends on the individual algorithm being accelerated and the dataset
being processed. As with cuML itself, you will generally see the most benefit
when ``cuml.accel`` is used on large datasets. Please see
`Zero Code Change Benchmarks <zero-code-change-benchmarks.rst>`_ for some representative benchmarks.

Please note that the first time an estimator method is called in a Python
process, there may be some overhead due to JIT compilation of cupy kernels. To
get an accurate sense of performance, run the method once on a small subset of
data before measuring runtime on a full-scale dataset.

6. Will I run out of GPU memory if I use ``cuml.accel``?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``cuml.accel`` will use CUDA `managed memory <https://developer.nvidia.com/blog/unified-memory-cuda-beginners/>`_ for allocations on NVIDIA GPUs. This means that host memory can be used to augment GPU memory, and data will be migrated automatically as necessary. This does not mean that ``cuml.accel`` is entirely impervious to OOM errors, however. Very large datasets can exhaust the entirety of both host and device memory. Additionally, if device memory is heavily oversubscribed, it can lead to slow execution. ``cuml.accel`` is designed to minimize both possibilities, but if you observe OOM errors or slow execution on data that should fit in combined host plus device memory for your system, please `report it <https://github.com/rapidsai/cuml/issues/new?template=bug_report.md>`_, and the RAPIDS team will investigate.

.. note::
   When running in Windows Subsystem for Linux 2 (WSL2), managed memory is not supported. Users may need to be more careful about memory management and consider using the ``--disable-uvm`` flag if experiencing memory-related issues.

7. What is the relationship between ``cuml.accel`` and ``cudf.pandas``?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Both projects serve a similar role. Just as ``cuml.accel`` offers zero code
change acceleration for Scikit-Learn and similar packages, ``cudf.pandas``
offers zero code change acceleration for Pandas.

Using them together is supported as an experimental feature. To do this from the
CLI, the flag ``--cudf-pandas`` can be added to the ``cuml.accel`` call:

.. code-block:: console

   python -m cuml.accel --cudf-pandas

For Jupyter notebooks, use the following approach to turn on both:

.. code-block::

   %load_ext cudf.pandas
   from cuml.experimental.accel import install
   install()

A single magic invocation is planned for a future release of cuML.


8. What happens if something in my script is not implemented in cuML?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``cuml.accel`` should cleanly and transparently fall back to the CPU
implementation for any methods or estimators which are not implemented in cuML.
If it does not do so, please `report it as a bug <https://github.com/rapidsai/cuml/issues/new?template=bug_report.md>`_, and the RAPIDS team will investigate.

9. I've discovered a bug in ``cuml.accel``. How do I report it?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Bugs affecting ``cuml.accel`` can be reported via the `cuML issue tracker <https://github.com/rapidsai/cuml/issues/new?template=bug_report.md>`_. If you observe a significant difference in the quality of output with and without ``cuml.accel``, please report it as a bug. These issues will be taken especially seriously. Similarly, if runtime slows down for your estimator when using ``cuml.accel``, the RAPIDS team will try to triage and fix the issue as soon as possible. Note that library import time *will* be longer when using ``cuml.accel``, so please exclude that from runtime. Long import time is a known issue and will be improved with subsequent releases of cuML.

10. If I serialize a model using ``cuml.accel``, can I load it without ``cuml.accel``?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This is a common use case for ``cuml.accel`` and cuML in general, since it may be useful to train
a model using NVIDIA GPUs but deploy it for inference in an environment that
does not have access to NVIDIA GPUs.

Currently, models serialized with
``cuml.accel`` need to be converted to pure Scikit-Learn (or UMAP/HDBSCAN/...).
After serializing a model, using either pickle or joblib, for example to `model_pickled.pkl`,
that model can then be converted to a regular sklearn/umap-learn/hdbscan pickled model with:

.. code-block:: console

    python -m cuml.accel --convert-to-sklearn model_pickled.pkl --format pickle --output converted_model.pkl


The `converted_model.pkl` is now a regular pickled/joblib serialized model,
that can be deserialized and used in a computer/environment without cuML or GPUs.

This conversion step should become unnecessary in a future release of cuML.
