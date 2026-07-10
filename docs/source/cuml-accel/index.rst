Zero Code Change Acceleration
=============================

``cuml.accel`` runs supported Scikit-Learn, UMAP, and HDBSCAN workloads on an
NVIDIA GPU without changing the Python code that uses those libraries. It is a
good fit when you want to accelerate an existing workflow, keep the familiar
APIs, or quickly evaluate the benefit of GPU acceleration before considering a
direct cuML migration.

Enable ``cuml.accel`` before importing the libraries you want to accelerate.
Your existing code then remains unchanged:

.. code-block:: python

   from sklearn.datasets import make_regression
   from sklearn.linear_model import Ridge

   X, y = make_regression(n_samples=1_000_000, random_state=0)
   model = Ridge().fit(X, y)
   predictions = model.predict(X)

Run a script through the ``cuml.accel`` command-line interface:

.. code-block:: console

   python -m cuml.accel script.py

Or load the extension at the top of a Jupyter notebook, before other imports:

.. code-block:: python

   %load_ext cuml.accel

``cuml.accel`` transparently falls back to the original CPU implementation
when an estimator or operation cannot be accelerated. Fallback can depend on
parameters, input types, methods, or library versions, so existing workflows
continue to run even when only part of a pipeline is GPU accelerated. Use the
:doc:`logging and profiling tools <understanding-acceleration>` to see exactly
where execution occurs.

.. _cuml-accel-env-var:

Other Activation Methods
------------------------

For third-party applications whose code you do not control, set
``CUML_ACCEL_ENABLED`` to ``1`` or ``true`` (case insensitive):

.. code-block:: console

   CUML_ACCEL_ENABLED=1 python script.py

This loads the accelerator for every Python program launched with the variable
set and may add startup overhead. If cuML is not installed correctly, this
environment variable is silently ignored, so the CLI or notebook extension is
usually easier to validate. See :doc:`examples/third-party-apps` for more.

You can also enable the accelerator programmatically before importing
Scikit-Learn, UMAP, or HDBSCAN:

.. code-block:: python

   import cuml

   cuml.accel.install()

Where to Go Next
----------------

* :doc:`supported-functionality` lists the estimators that can be accelerated.
* :doc:`limitations` documents estimator-specific fallback conditions and
  behavioral differences.
* :doc:`understanding-acceleration` explains fallback, results, performance,
  memory management, logging, and profiling.
* :doc:`benchmarks` provides representative performance results.
* :doc:`examples/index` contains complete examples and notebook workflows.
* :doc:`faq` covers interoperability, serialization, and bug reporting.

.. toctree::
   :hidden:

   self
   supported-functionality
   limitations
   understanding-acceleration
   benchmarks
   examples/index
   faq
