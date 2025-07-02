Understanding Acceleration and Fallbacks with Logging
====================================================

``cuml.accel`` provides comprehensive logging to help you understand which
operations are being accelerated on GPU and which are falling back to CPU
execution. This can be particularly useful for debugging performance issues
or understanding why certain operations might not be accelerated.

To enable logging, you can set the logging level in several ways:

**Command Line Interface (CLI):**

When running scripts with ``cuml.accel``, you can use the ``-v`` or ``--verbose`` flag:

.. code-block:: console

   # Show warnings only (default)
   python -m cuml.accel myscript.py

   # Show info level logs
   python -m cuml.accel -v myscript.py

   # Show debug level logs (most verbose)
   python -m cuml.accel -vv myscript.py

**Programmatic Installation:**

When using the programmatic installation method, you can set the log level directly:

.. code-block:: python

   from cuml.accel import install
   from cuml.internals import logger

   # Install with debug logging
   install(log_level=logger.level_enum.debug)

   # Install with info logging
   install(log_level=logger.level_enum.info)

   # Install with warning logging (default)
   install(log_level=logger.level_enum.warn)

**Jupyter Notebooks:**

Since the magic command doesn't accept arguments, use the programmatic installation:

.. code-block::

   from cuml.accel import install
   from cuml.internals import logger

   # Install with desired log level before other imports
   install(log_level=logger.level_enum.debug)

The logging system provides different levels of detail:

* **WARN level**: Shows only warnings about failed accelerations and fallbacks
* **INFO level**: Shows successful accelerations and important fallbacks
* **DEBUG level**: Shows detailed information about GPU initialization,
  parameter synchronization, attribute synchronization, and all method calls

Examples for log messages you might see:

**Successful GPU Acceleration:**
   - ``[cuml.accel] Initialized estimator 'Ridge' for GPU acceleration``
   - ``[cuml.accel] Successfully accelerated 'Ridge.fit()' call``
   - ``[cuml.accel] Successfully accelerated 'Ridge.predict()' call``

**Parameter and Attribute Synchronization:**
   - ``[cuml.accel] Synced parameters from CPU to GPU for 'Ridge'``
   - ``[cuml.accel] Synced fit attributes from GPU to CPU for 'Ridge'``

**Fallbacks to CPU:**
   - ``[cuml.accel] Failed to initialize 'KMeans' with GPU acceleration``
   - ``[cuml.accel] Failed to accelerate 'Ridge.fit()': Multioutput `y` is not supported - falling back to CPU``
   - ``[cuml.accel] Unable to accelerate 'Ridge.predict()' call: Sparse inputs are not supported``
   - ``[cuml.accel] Executing 'Ridge.predict()' on CPU``

**Programmatic Example:**

.. code-block:: python

   from sklearn.linear_model import Ridge
   import numpy as np

   # Create and fit a Ridge estimator
   ridge = Ridge(alpha=1.0)
   X = np.random.randn(100, 10)
   y = np.random.randn(100)

   # This will show initialization and successful acceleration logs
   ridge.fit(X, y)

   # This will show successful prediction acceleration
   predictions = ridge.predict(X)

Executing this with `python -m cuml.accel -v my_ml_script.py` will show the following output:

.. code-block:: console

   [cuml.accel] Initialized estimator 'Ridge' for GPU acceleration
   [cuml.accel] Successfully accelerated 'Ridge.fit()' call
   [cuml.accel] Successfully accelerated 'Ridge.predict()' call

This logging information can help you:

* Identify which parts of your pipeline are being accelerated
* Understand why certain operations fall back to CPU
* Debug performance issues by seeing where GPU acceleration fails
* Optimize your code by understanding synchronization patterns
