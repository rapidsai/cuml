FAQ
---

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
memory usage. ``cuml.accel`` will enable `unified or managed memory
<https://developer.nvidia.com/blog/unified-memory-cuda-beginners/>`_ (provided
the platform supports it and `rmm
<https://docs.rapids.ai/api/rmm/stable/guide/>`_ hasn't already been configured).
Using managed memory can help reduce the risk of CUDA out-of-memory errors.
In contrast, cuML defaults to ordinary device memory, which can offer improved
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
    * ``sklearn.manifold.SpectralEmbedding``
    * ``sklearn.neighbors.NearestNeighbors``
    * ``sklearn.neighbors.KNeighborsClassifier``
    * ``sklearn.neighbors.KNeighborsRegressor``
    * ``sklearn.svm.SVC``
    * ``sklearn.svm.SVR``
    * ``sklearn.svm.LinearSVC``
    * ``sklearn.svm.LinearSVR``

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

When possible, ``cuml.accel`` will enable `managed memory
<https://developer.nvidia.com/blog/unified-memory-cuda-beginners/>`_ for
allocations on NVIDIA GPUs. This means that host memory can be used to augment
GPU memory, and data will be migrated automatically as necessary. This does not
mean that ``cuml.accel`` is entirely impervious to OOM errors, however. Very
large datasets can exhaust the entirety of both host and device memory.
Additionally, if device memory is heavily oversubscribed, it can lead to slow
execution. ``cuml.accel`` is designed to minimize both possibilities, but if
you observe OOM errors or slow execution on data that should fit in combined
host plus device memory for your system, please `report it
<https://github.com/rapidsai/cuml/issues/new?template=bug_report.md>`_, and the
RAPIDS team will investigate.

.. note::

   Managed memory will not be enabled:

   - When running in Windows Subsystem for Linux 2 (WSL2), where it's not
     supported.
   - When `rmm <https://docs.rapids.ai/api/rmm/stable/guide/>`_ is already
     configured externally to `cuml.accel`.

   Users in these situations may need to be more cognizant about their GPU
   memory usage to ensure they don't exceed the memory capacity of their GPU.

7. What is the relationship between ``cuml.accel`` and ``cudf.pandas``?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Both projects serve a similar role. Just as ``cuml.accel`` offers zero code
change acceleration for Scikit-Learn and similar packages, ``cudf.pandas``
offers zero code change acceleration for Pandas.

Using them together is supported. To do this from the CLI, both accelerators
may be invoked like:

.. code-block:: console

   python -m cudf.pandas -m cuml.accel ...

For Jupyter notebooks, use the following approach to turn on both:

.. code-block::

   %load_ext cudf.pandas
   %load_ext cuml.accel


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
This is a common use case for ``cuml.accel`` and cuML in general, since it may
be useful to train a model using NVIDIA GPUs but deploy it for inference in an
environment that does not have access to NVIDIA GPUs.

Models serialized with ``cuml.accel`` may be loaded in environments without
``cuml.accel`` - in this case they'll be loaded as their normal
sklearn/umap-learn/hdbscan counterpart.

Note that the same serialized model may also be loaded with ``cuml.accel``
active, in which case they'll be accelerated ``cuml.accel`` backed models.

11. How can I tell which parts of my code are being accelerated and why some operations might not be?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``cuml.accel`` provides comprehensive logging that shows you exactly what's happening
with your code. You can enable logging to see which operations are successfully
accelerated on GPU and which fall back to CPU execution.

**To enable logging:**

* **CLI**: Use the ``-v`` flag for info level or ``-vv`` for debug level:
  ``python -m cuml.accel -v myscript.py``
* **Programmatic**: Use the ``cuml.accel.install()`` function with a log level:
  ``install(log_level="info")``

For detailed information about logging and troubleshooting, see
:doc:`logging-and-profiling`.
