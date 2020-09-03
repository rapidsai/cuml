Intro and key concepts for cuML
=================================

cuML accelerates machine learning on GPUs. The library follows a
couple of key principals, and understanding these will help you take
full advantage cuML.


1. Where possible, match the scikit-learn API
---------------------------------------------

cuML estimators look and feel just like `scikit-learn estimators
<https://scikit-learn.org/stable/developers/develop.html>`_. You
initialize them with key parameters, fit them with a ``fit`` method,
then call ``predict`` or ``transform`` for inference.


.. code-block:: python

   import cuml.LinearRegression
   
   model = cuml.LinearRegression()
   model.fit(X_train, y)
   y_prediction = model.predict(X_test)

You can find many more complete examples in the `Introductory Notebook
<estimator_intro.ipynb>`_ and in the cuML API documentation.

2. Accept flexible input types, return predictable output types
---------------------------------------------------------------

cuML estimators can accept NumPy arrays, cuDF dataframes, cuPy arrays,
2d PyTorch tensors, and really any kind of standards-based Python
array input you can throw at them. This relies on the ``__array__``
and ``__cuda_array_interface__`` standards, widely used throughout the
PyData community.

By default, outputs will mirror the data type you provided. So, if you
fit a model with a NumPy array, the ``model.coef_`` property
containing fitted coefficients will also be a NumPy array. If you fit
a model using cuDF's GPU-based DataFrame and Series objects, the
model's output properties will be cuDF objects. You can always
override this behavior and select a default datatype with the
`memory_utils.set_global_output_type
<https://docs.rapids.ai/api/cuml/nightly/api.html#datatype-configuration>`_
function.

The `RAPIDS Configurable Input and Output Types
<https://medium.com/@dantegd/e719d72c135b>`_ blog post goes into much
more detail explaining this approach.

3. Be fast!
-----------

cuML's estimators rely on highly-optimized CUDA primitives and
algorithms within ``libcuml``. On a modern GPU, these can exceed the
performance of CPU-based equivalents by a factor of anything from 4x
(for a medium-sized linear regression) to over 1000x (for large-scale
tSNE dimensionality reduction). The `cuml.benchmark
<https://docs.rapids.ai/api/cuml/nightly/api.html#benchmarking>`_ module
provides an easy interface to benchmark your own hardware.

To maximize performance, keep in mind - a modern GPU can have over
5000 cores, so make sure you're providing enough data to keep it busy!
In many cases, performance advantages appear as the dataset grows.


Learn more
----------

To get started learning cuML, walk through the `Introductory Notebook
<estimator_intro.ipynb>`_. Then try out some of the other notebook
examples in the ``notebooks`` directory of the repository. Finally, do
a deeper dive with the `cuML blogs <cuml_blogs.rst>`_.
