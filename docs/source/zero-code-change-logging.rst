Logging and Debugging
=====================

``cuml.accel`` provides comprehensive logging to help you understand which
operations are being accelerated on GPU and which are falling back to CPU
execution. This can be particularly useful for debugging performance issues
or understanding why certain operations might not be accelerated.

To enable logging, you can set the logging level in several ways:

Command Line Interface (CLI)
----------------------------

When running scripts with ``cuml.accel``, you can use the ``-v`` or ``--verbose`` flag:

.. code-block:: console

   # Show warnings only (default)
   python -m cuml.accel myscript.py

   # Show info level logs
   python -m cuml.accel -v myscript.py

   # Show debug level logs (most verbose)
   python -m cuml.accel -vv myscript.py

Programmatic Installation
-------------------------

When using the programmatic installation method, you can set the log level directly:

.. code-block:: python

   from cuml.internals import logger

   # Install with debug logging
   install(log_level="debug")

   # Install with info logging
   install(log_level="info")

   # Install with warning logging (default)
   install(log_level="warn")

Jupyter Notebooks
-----------------

Since the magic command doesn't accept arguments, use the programmatic installation:

.. code-block::

   from cuml.accel import install

   # Install with desired log level before other imports
   install(log_level="debug")

The logging system provides different levels of detail:

* **WARN level**: Shows only warnings and errors, the default level
* **INFO level**: Shows information on where dispatched methods ran (GPU or CPU) and why
* **DEBUG level**: Shows more detailed information about GPU initialization,
  parameter synchronization, attribute synchronization, and all method calls

Example
-------

.. code-block:: python

   from sklearn.linear_model import Ridge
   from sklearn.datasets import make_regression

   X, y = make_regression()

   # Fit and predict on GPU
   ridge = Ridge(alpha=1.0)
   ridge.fit(X, y)
   ridge.predict(X)

   # Retry, using a hyperparameter that isn't supported on GPU
   ridge = Ridge(positive=True)
   ridge.fit(X, y)
   ridge.predict(X)


Executing this with ``python -m cuml.accel -v script.py`` will show the following output:

.. code-block:: console

   [cuml.accel] Accelerator installed.
   [cuml.accel] `Ridge.fit` ran on GPU
   [cuml.accel] `Ridge.predict` ran on GPU
   [cuml.accel] `Ridge.fit` falling back to CPU: `positive=True` is not supported
   [cuml.accel] `Ridge.fit` ran on CPU
   [cuml.accel] `Ridge.predict` ran on CPU

This logging information can help you:

* Identify which parts of your pipeline are being accelerated
* Understand why certain operations fall back to CPU
* Debug performance issues by seeing where GPU acceleration fails
* Optimize your code by understanding synchronization patterns
