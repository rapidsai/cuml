Welcome to cuML's documentation!
=================================

cuML is a suite of fast, GPU-accelerated machine learning algorithms
designed for data science and analytical tasks. Our API mirrors scikit-learn,
and we provide practitioners with the easy fit-predict-transform paradigm
without ever having to program on a GPU. With `cuml.accel`, cuML can also
automatically accelerate existing code with zero code changes.

cuML delivers **10-50x faster performance** than CPU-based alternatives for
realistic workloads and supports **50+ algorithms** across all major machine
learning categories, including clustering, regression, classification,
dimensionality reduction, and time series analysis. With comprehensive
**multi-GPU and multi-node support** via Dask, cuML scales from single
workstations to large clusters.

As data gets larger, algorithms running on a CPU become slow and cumbersome.
RAPIDS provides users a streamlined approach where data is initially loaded
in the GPU, and compute tasks can be performed on it directly.

Quick Start
===========

.. code-block:: python

   from cuml.datasets import make_blobs
   from cuml.cluster import DBSCAN

   # Create sample data
   X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

   # Fit clustering model
   dbscan = DBSCAN(eps=1.0, min_samples=5)
   dbscan.fit(X)
   print(dbscan.labels_)

Key Features
============

* **GPU Acceleration**: 10-50x faster than CPU-based alternatives
* **Scikit-learn Compatible**: Drop-in replacement for most sklearn algorithms
* **Multi-GPU Support**: Scale across multiple GPUs and nodes with Dask
* **Comprehensive Coverage**: 50+ algorithms across all major ML categories
* **Flexible Input**: Works with NumPy, cuDF, cuPy, and PyTorch tensors
* **Production Ready**: Battle-tested in enterprise environments

Installation
============

cuML is available through conda and pip. For detailed installation instructions,
visit the `RAPIDS Release Selector <https://docs.rapids.ai/install#selector>`_.

.. note::
   cuML is only supported on Linux operating systems and WSL 2. See
   `the RAPIDS install page <https://docs.rapids.ai/install/#system-req>`_
   for details on system and hardware requirements.

Part of RAPIDS
==============

cuML is part of the RAPIDS suite of open source libraries that enable
end-to-end data science and analytics pipelines entirely on GPUs. It works
seamlessly with other RAPIDS libraries like cuDF for data manipulation and
cuGraph for graph analytics.

Community & Support
===================

* :doc:`User Guide <user_guide>` - Comprehensive usage documentation
* :doc:`API Reference <api>` - Complete API documentation
* `GitHub Issues <https://github.com/rapidsai/cuml/issues>`_ - Report bugs and request features
* `RAPIDS Community <https://rapids.ai/community.html>`_ - Join our community

Documentation
=============

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   cuml_intro.rst
   user_guide.rst

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api.rst
   cuml-accel/index.rst

.. toctree::
   :maxdepth: 2
   :caption: Additional Resources

   FIL.rst
   cuml_blogs.rst
