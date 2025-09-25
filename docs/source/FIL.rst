FIL - RAPIDS Forest Inference Library
=====================================

The Forest Inference Library is a subset of cuML designed to accelerate inference for tree-based models regardless of what framework they are trained on. FIL can accelerate XGBoost models, Scikit-Learn/cuML ``RandomForest`` models, LightGBM models, and any other model that can be converted to Treelite. An example invocation is shown below:

.. code-block:: python

    from cuml import ForestInference

    fil_model = ForestInference.load("./my_xgboost_classifier.ubj", is_classifier=True)
    class_predictions = fil_model.predict(input_data)

FIL typically offers speedups of 80x or more relative to native inference with e.g. a Scikit-Learn ``RandomForest`` model on CPU.

Optional CPU Execution
----------------------
While FIL offers the most benefit for large models and batch sizes by taking advantage of the speed and parallelism of NVIDIA GPUs, it can also be used to speed up inference on CPUs. This can be convenient for testing in environments without access to GPUs. It can also be useful for deployments which experience dramatic shifts in traffic. When the number of incoming inference requests is low, CPU execution can be used. When traffic spikes, the deployment can seamlessly scale up onto GPUs in order to handle the additional load as cheaply as possible without significantly increasing latency.

Optimizing Hyperparameters
--------------------------
FIL has a number of performance hyperparameters which can be used to get the maximum performance for a specific model and batch size. These can be tuned manually, but the built-in ``.optimize`` method makes it easy to quickly set those hyperparameters to the optimal value for a specific use case:

.. code-block:: python

    fil_model.optimize(batch_size=1_000_000)
    output = fil_model.predict(input_data)

This method will optimize the ``layout`` hyperparameter, which determines the order in which tree nodes are arranged in memory as well as ``default_chunk_size``, which determines the granularity of parallelization during inference.

Additionally, you may wish to experiment with the ``align_bytes`` parameter. Its default value is typically close enough to optimal that it is not automatically searched during auto-optimization, but to squeeze the most performance possible out of FIL, try either 0 or 128 on GPU and 0 or 64 on CPU.

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

Extra Prediction Modes
----------------------
To gain additional insight on how models arrive at their inference decision, FIL now includes the ``.predict_per_tree`` and ``.apply`` methods. The first returns the output for every single tree in the ensemble individually. The second returns the ID of the leaf node obtained for every tree in the ensemble.

Upcoming Changes
----------------
In RAPIDS 25.06, the shape of output arrays will change slightly for some models. Binary classifiers will return an array of solely the probabilities of the positive class for ``predict_proba`` calls. This both reduces memory requirements and improves performance. To convert to the old format, the following snippet can be used:

.. code-block:: python

    import numpy as np  # Use cupy or numpy depending on which you use for input data

    out = fil_model.predict_proba(input_data)
    # Starting in RAPIDS 25.06, the following can be used to obtain the old output shape
    out = np.stack([1 - out, out], axis=1)

Additionally, ``.predict`` calls will output two-dimensional arrays beginning in 25.06. This is in preparation for supporting multi-target regression and classification models. The old shape can be obtained via the following snippet:

.. code-block:: python

    import numpy as np  # Use cupy or numpy depending on which you use for input data

    out = fil_model.predict(input_data)
    # Starting in RAPIDS 25.06, the following can be used to obtain the old output shape
    out = out.flatten()

To use these new behaviors immediately, the ``ForestInference`` estimator can be imported from the ``experimental`` namespace:

.. code-block:: python

    from cuml.experimental.fil import ForestInference
