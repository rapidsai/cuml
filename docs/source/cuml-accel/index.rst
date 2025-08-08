Zero Code Change Acceleration
=============================

The ``cuml.accel`` zero code change accelerator provides a mechanism to
accelerate existing python machine learning code on the GPU, *without requiring
any changes to that code*. Depending on the data size and algorithms chosen,
this may result in :doc:`major speedups <benchmarks>`.

.. code-block:: python

    %%load_ext cuml.accel
    # Certain operations in common ML libraries (sklearn, umap, hdbscan)
    # are now GPU accelerated

    from sklearn.datasets import make_regression
    from sklearn.linear_model import ElasticNet

    X, y = make_regression(n_samples=1_000_000)

    model = ElasticNet()
    model.fit(X, y)   # runs on GPU!
    model.predict(X)  # runs on GPU!

Currently ``cuml.accel`` targets ``sklearn``, ``umap``, and ``hdbscan`` as
libraries to accelerate. Functionality that isn't yet supported will fallback
to CPU execution. See :doc:`limitations` for more information on what's
currently accelerated and what requires a CPU fallback.

Usage
-----

``cuml.accel`` comes standard with ``cuml``, no additional installation
requirements are needed. It's designed to be used with existing code that makes
use of ``sklearn``, ``umap`` or ``hdbscan``, with the only change being something
to enable the use of the accelerator.

Command Line Interface (CLI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When executing from the commandline, you can use ``python -m cuml.accel`` in
place of ``python`` to execute python code with the accelerator enabled.

.. code-block:: console

   python -m cuml.accel script.py


Jupyter/IPython
~~~~~~~~~~~~~~~

The same functionality is available in Jupyter notebooks or IPython by
executing the following cell magic at the top (before other imports):

.. code-block::

   %%load_ext cuml.accel

You can see an example of this in :doc:`this example
<examples/plot_kmeans_digits>`.

Environment Variable
~~~~~~~~~~~~~~~~~~~~

The accelerator may also be enabled by setting the ``CUML_ACCEL_ENABLED``
environment variable to ``1`` or ``true`` (case insensitive).

.. code-block:: console

   # Define it just for a single command
   CUML_ACCEL_ENABLED=1 python script.py

   # Or set it to persist in your current shell session
   export CUML_ACCEL_ENABLED=1

Note that any python program running with the environment defined this way
will load the accelerator, which may result in a measurable startup overhead.

Additionally, if ``cuml`` is not installed properly in your environment, the
``CUML_ACCEL_ENABLED`` environment variable will be silently ignored (and
normal CPU execution will occur). For this reason one of the other methods
listed here may be preferred, as failure to invoke the accelerator will result
in a detectable error.

Enabling Programmatically
~~~~~~~~~~~~~~~~~~~~~~~~~

When needed, the accelerator may also be enabled programmatically by calling
`cuml.accel.install`. Note that you'll want to call this early in your code,
before importing functionality from ``sklearn``/``umap``/``hdbscan``.

.. code-block:: python

   import cuml
   cuml.accel.install()

.. toctree::
   :hidden:

   self
   logging-and-profiling.rst
   limitations.rst
   faq.rst
   benchmarks.rst
   examples/index.rst
