Supported Versions
==================

Please see https://docs.rapids.ai/install/ for RAPIDS-wide version support.

We aim to meet the `SPEC 0 guidelines <https://scientific-python.org/specs/spec-0000/>`_ for minimal supported versions.

Required Runtime Dependencies
---------------------------

The following dependencies are required for the cuML library:

* **NumPy**: >=1.23,<3.0a0
* **scikit-learn**: >=1.4
* **scipy**: >=1.8.0
* **numba**: >=0.59.1,<0.62.0a0
* **cupy**: cupy-cuda12x>=12.0.0
* **treelite**: ==4.4.1


Optional Runtime Dependencies
---------------------------

The following dependencies are optional and provide additional functionality:

* **xgboost**: >=2.1.0 (for gradient boosting algorithms)
* **hdbscan**: >=0.8.39,<0.8.40 (for hierarchical density-based clustering)
* **umap-learn**: ==0.5.7 (for dimensionality reduction)
* **pynndescent**: (for approximate nearest neighbor search)

RAPIDS Dependencies
------------------

cuML dependencies within the RAPIDS ecosystem are pinned to the same version. For example, cuML 25.08 is compatible with and only with cuDF 25.08, cuVS 25.08, and other RAPIDS libraries at version 25.08.
