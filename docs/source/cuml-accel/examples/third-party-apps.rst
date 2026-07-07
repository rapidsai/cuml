Accelerating Third-Party Applications
======================================

The ``CUML_ACCEL_ENABLED`` environment variable lets you GPU-accelerate any
Python application that uses ``sklearn``, ``umap``, or ``hdbscan``.
Even applications whose code you cannot modify. This is useful for
installed CLI tools, applications, and third-party libraries.

.. code-block:: console

   CUML_ACCEL_ENABLED=1 some-third-party-tool [args...]

When :ref:`CUML_ACCEL_ENABLED=1 is defined <cuml-accel-env-var>`,
`cuml.accel` will be enabled as part of the normal Python interpreter
startup, letting you accelerate Python applications without modification

This means you do not need access to an application's source code: set the
environment variable and the acceleration applies automatically.

Example: Embedding Visualization with embedding-atlas
-----------------------------------------------------

`embedding-atlas <https://github.com/apple/embedding-atlas>`_ is Apple's
open-source tool for interactive visualization of large embedding datasets.
Given a text dataset, it computes sentence embeddings, projects them to 2D
using `UMAP <https://umap-learn.readthedocs.io/>`_, and launches a
browser-based explorer.

Install it alongside ``cuml``:

.. code-block:: console

   pip install embedding-atlas

Run it on a Hugging Face dataset. The example below uses
`TinyStories <https://huggingface.co/datasets/roneneldan/TinyStories>`_,
a dataset of 2M+ short stories:

.. code-block:: console

   # CPU -- UMAP runs on CPU
   embedding-atlas roneneldan/TinyStories --text text \
       --split train --sample 1000000

   # GPU -- set environment variable; no other changes needed
   CUML_ACCEL_ENABLED=1 embedding-atlas roneneldan/TinyStories --text text \
       --split train --sample 1000000

The only change between the two commands is the environment variable.
``embedding-atlas`` computes embeddings with sentence-transformers (which
already uses the GPU), then runs UMAP for dimensionality reduction.
``cuml.accel`` intercepts the ``umap.UMAP`` call inside ``embedding-atlas``
and dispatches ``fit_transform`` to cuML's GPU implementation.

Use a smaller ``--sample`` value (e.g. 250000) for a quicker test run.
The UMAP speedup grows with dataset size.

To confirm GPU dispatch, add ``CUML_ACCEL_LOG_LEVEL=info``:

.. code-block:: console

   CUML_ACCEL_ENABLED=1 CUML_ACCEL_LOG_LEVEL=info embedding-atlas \
       roneneldan/TinyStories --text text --split train --sample 1000000

You should see the following messages amongst the other output:

.. code-block:: text

   [cuml.accel] Accelerator installed.
   [cuml.accel] `UMAP.fit_transform` ran on GPU

Results
~~~~~~~

At the time of writing and on the hardware the author used the
``fit_transform`` step saw a roughly **~4x speedup** because cuML's GPU
UMAP replaces the CPU optimization. The KNN step (``nearest_neighbors``)
is a standalone function call that ``cuml.accel`` does not currently
intercept, so it runs on CPU in both cases. Despite this, the overall
UMAP step is still **~2x faster**.

At smaller scales (< 100K rows) the UMAP step is already fast on CPU and
the speedup is less pronounced. The benefit grows with dataset size.


Identifying Acceleratable Applications
---------------------------------------

Any Python tool that calls one of the following is a candidate for
``CUML_ACCEL_ENABLED``:

- ``sklearn`` estimators (KMeans, PCA, DBSCAN, RandomForest,
  LogisticRegression, NearestNeighbors, and
  :doc:`many more <../faq>`)
- ``umap.UMAP``
- ``hdbscan.HDBSCAN``

A quick way to check: search an application's dependencies for
``scikit-learn``, ``umap-learn``, or ``hdbscan``, or run with
``CUML_ACCEL_LOG_LEVEL=info`` and look for ``ran on GPU`` messages
in the output.

Checking for CPU Fallbacks
--------------------------

Not all parameter combinations are supported on the GPU. When
``cuml.accel`` encounters an unsupported configuration, it silently
falls back to CPU execution. To detect this, set the log level to
``info`` or ``debug``:

.. code-block:: console

   CUML_ACCEL_ENABLED=1 CUML_ACCEL_LOG_LEVEL=info python app.py

Lines containing ``ran on GPU`` confirm GPU execution. Lines
containing ``falling back to CPU`` indicate a fallback, along with
the reason. See :doc:`../logging-and-profiling` for more detail.
