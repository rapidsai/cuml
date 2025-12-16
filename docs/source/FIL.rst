FIL - RAPIDS Forest Inference Library
=====================================

The Forest Inference Library (FIL) is a component of cuML, providing a
high-performance inference engine designed to accelerate tree-based machine
learning models on both GPU and CPU. FIL delivers significant speedups over
traditional CPU-based inference while maintaining compatibility with models
trained in popular frameworks.

**Key Benefits:**

- FIL typically offers a speedup of 80x or more over scikit-learn native execution
- Support for XGBoost, Scikit-Learn, LightGBM, and Treelite-compatible models
- Seamless GPU/CPU execution switching
- Built-in auto-optimization for maximum performance
- Advanced inference APIs for granular tree analysis

**Quick Start:**

.. code-block:: python

    import xgboost as xgb
    import numpy as np
    from cuml.fil import ForestInference

    # Train your model as usual and save it
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)
    xgb_model.save_model("xgb_model.ubj")

    # Load into FIL and auto-tune for your batch size
    fil_model = ForestInference.load("xgb_model.ubj", is_classifier=True)
    fil_model.optimize(batch_size=1024)

    # Now you can predict with FIL directly
    predictions = fil_model.predict(X_test)
    probabilities = fil_model.predict_proba(X_test)

Performance Optimization
-------------------------
FIL includes built-in auto-optimization that automatically tunes performance hyperparameters for your specific model and batch size, eliminating the need for manual tuning in most cases:

.. code-block:: python

    fil_model = ForestInference.load("model.ubj", is_classifier=True)
    fil_model.optimize(batch_size=1_000_000)

    # Check which hyperparameters were selected
    print(f"Layout: {fil_model.layout}")
    print(f"Chunk size: {fil_model.default_chunk_size}")

    result = fil_model.predict(data)

The optimization process tests different memory layouts and chunk sizes to find the optimal configuration for your specific use case.

**Key Hyperparameters:**

- ``layout``: Determines the order in which tree nodes are arranged in memory (depth_first, layered, breadth_first)
- ``default_chunk_size``: Controls the granularity of parallelization during inference
- ``align_bytes``: Cache line alignment for optimal memory access patterns

**Manual Tuning:**
For advanced users, you can experiment with the ``align_bytes`` parameter. Its default value is typically close enough to optimal that it is not automatically searched during auto-optimization, but to squeeze the most performance possible out of FIL, try either 0 or 128 on GPU and 0 or 64 on CPU.

Optional CPU Execution
----------------------
While FIL offers the most benefit for large models and batch sizes by taking advantage of the speed and parallelism of NVIDIA GPUs, it can also be used to speed up inference on CPUs. This can be convenient for testing in environments without access to GPUs. It can also be useful for deployments which experience dramatic shifts in traffic. When the number of incoming inference requests is low, CPU execution can be used. When traffic spikes, the deployment can seamlessly scale up onto GPUs in order to handle the additional load as cheaply as possible without significantly increasing latency.

You can use FIL in CPU mode with a context manager:

.. code-block:: python

    from cuml.fil import ForestInference, set_fil_device_type

    with set_fil_device_type("cpu"):
        fil_model = ForestInference.load("xgboost_model.ubj")
        result = fil_model.predict(data)

Advanced Prediction APIs
-------------------------
FIL includes advanced prediction methods that provide granular information about individual trees in the ensemble, enabling novel ensembling techniques and analysis:

**Per-Tree Predictions**
The ``.predict_per_tree`` method returns the output of every single tree individually:

.. code-block:: python

    per_tree = fil_model.predict_per_tree(X)
    mean = per_tree.mean(axis=1)
    lower = np.percentile(per_tree, 10, axis=1)
    upper = np.percentile(per_tree, 90, axis=1)

This enables advanced techniques like:

- Weighted voting based on tree age, out-of-bag AUC, or data-drift scores
- Prediction intervals without bootstrapping
- Novel ensembling techniques with no retraining required

**Leaf Node Analysis**
The ``.apply`` method returns the leaf node ID for every tree, enabling similarity analysis:

.. code-block:: python

    leaf = fil_model.apply(X)
    sim = (leaf[i] == leaf[j]).mean()  # fraction of matching leaves
    print(f"{sim:.0%} of trees agree on rows {i} & {j}")

This opens forest models to novel uses beyond straightforward regression or classification, such as measuring data similarity and understanding model behavior.

Use Cases
---------
FIL is ideal for many scenarios:

**High-Performance Applications:**

- User-facing APIs where every millisecond counts
- High-volume batch jobs (ad-click scoring, IoT analytics)
- Real-time inference with sub-10ms latency requirements

**Flexible Deployment:**

- Hybrid deployments - same model file, choose CPU or GPU at runtime
- Prototype locally and deploy to GPU-accelerated production servers
- Scale down to CPU-only machines during light traffic, scale up with GPUs during peak loads

**Cost Optimization:**

- One GPU can replace CPUs with 50+ cores
- Significant cost reduction for high-throughput inference workloads
- Efficient resource utilization across different traffic patterns

**Advanced Analytics:**

- Novel ensembling techniques with per-tree analysis
- Data similarity measurement and model interpretability
- Prediction intervals and uncertainty quantification

API Reference
=============

See the :doc:`API reference <api>` for the API documentation.

Migration Guide
===============

FIL Redesign in RAPIDS 25.04
-----------------------------
FIL was completely redesigned in RAPIDS 25.04 with a new C++ implementation that provides significant performance improvements and new features:

**Key Changes in 25.04:**

- New C++ implementation for batched inference on GPU and CPU
- Built-in auto-optimization with ``.optimize()`` method
- Advanced inference APIs (``.predict_per_tree``, ``.apply``)
- Up to 4x faster GPU throughput than previous versions
- Enhanced memory layouts and cache optimization
- New parameter structure (``layout``, ``align_bytes``)
- Moved ``threshold`` from ``.load()`` to ``.predict()``

Migration from RAPIDS 25.04 to 25.06 (Output Shape Changes)
-----------------------------------------------------------
In RAPIDS 25.06, the shape of output arrays changed for some models. Binary classifiers now return an array of solely the probabilities of the positive class for ``predict_proba`` calls. This both reduces memory requirements and improves performance. To convert to the old format, the following snippet can be used:

.. code-block:: python

    import numpy as np  # Use cupy or numpy depending on which you use for input data

    out = fil_model.predict_proba(input_data)
    # Starting in RAPIDS 25.06, the following can be used to obtain the old output shape
    out = np.stack([1 - out, out], axis=1)

Additionally, ``.predict`` calls now output two-dimensional arrays beginning in 25.06. This is in preparation for supporting multi-target regression and classification models. The old shape can be obtained via the following snippet:

.. code-block:: python

    import numpy as np  # Use cupy or numpy depending on which you use for input data

    out = fil_model.predict(input_data)
    # Starting in RAPIDS 25.06, the following can be used to obtain the old output shape
    out = out.flatten()

To use these new behaviors immediately, the ``ForestInference`` estimator can be imported from the ``experimental`` namespace:

.. code-block:: python

    from cuml.experimental.fil import ForestInference

Migration from RAPIDS 24.12 to 25.04
------------------------------------

**Before (RAPIDS 24.12):**

.. code-block:: python

    fil_model = ForestInference.load(
        "./model.ubj",
        is_classifier=True,
        algo='TREE_REORG',  # Deprecated
        threshold=0.5,      # Now moved to predict()
        storage_type='DENSE'  # Deprecated
    )
    predictions = fil_model.predict(data)

**After (RAPIDS 25.04):**

.. code-block:: python

    fil_model = ForestInference.load(
        "./model.ubj",
        is_classifier=True,
        layout='depth_first'  # New parameter
    )
    predictions = fil_model.predict(data, threshold=0.5)  # threshold moved here

Deprecated ``load`` Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
As of RAPIDS 25.04, the following hyperparameters accepted by the ``.load`` method of previous versions of FIL have been deprecated.

- ``threshold`` (will trigger a deprecation warning if used; pass to ``.predict`` instead)
- ``algo`` (ignored, but a warning will be logged)
- ``storage_type`` (ignored, but a warning will be logged)
- ``blocks_per_sm`` (ignored, but a warning will be logged)
- ``threads_per_tree`` (ignored, but a warning will be logged)
- ``n_items`` (ignored, but a warning will be logged)
- ``compute_shape_str`` (ignored, but a warning will be logged)

New ``load`` Parameters
^^^^^^^^^^^^^^^^^^^^^^^
As of RAPIDS 25.04, the following new hyperparameters can be passed to the ``.load`` method

- ``layout``: Replaces the functionality of ``algo`` and specifies the in-memory layout of nodes in FIL forests. One of ``'depth_first'`` (default), ``'layered'`` or ``'breadth_first'``.
- ``align_bytes``: If specified, trees will be padded such that their in-memory size is a multiple of this value. This can sometimes improve performance by guaranteeing that memory reads from trees begin on a cache line boundary.

New Prediction Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^
As of RAPIDS 25.04, all prediction methods accept a ``chunk_size`` parameter, which determines how batches are further subdivided for parallel processing. The optimal value depends on hardware, model, and batch size, and it is difficult to predict in advance. Typically, it is best to use the ``.optimize`` method to determine the best chunk size for a given batch size. If ``chunk_size`` must be set manually, the only general rule of thumb is that larger batch sizes generally benefit from larger chunk sizes. On GPU, ``chunk_size`` can be any power of 2 from 1 to 32. On CPU, ``chunk_size`` can be any power of 2, but values above 512 rarely offer any benefit.

Additionally, ``threshold`` has been converted from a ``.load`` parameter to a ``.predict`` parameter.
