# Experimental FIL - RAPIDS Forest Inference Library

This experimental feature offers a new implementation of cuML's existing
Forest Inference Library. The primary advantages of this new
implementation are:

1. Models can now be evaluated on CPU in addition to GPU.
2. Faster GPU execution on some models and hardware.
3. Support for a wider range of Treelite's available model parameters.

In addition, there are a few limitations of this implementation,
including:

1. Models with shallow trees (depth 2-4) typically execute slower than with
   existing FIL.
2. This implementation has not been as exhaustively tested as the existing
   FIL.

If you need to absolutely maximize runtime performance, it is
recommended that you test both the new and existing FIL implementations with
realistic batch sizes on your target hardware to determine which is optimal
for your specific model. Generally, however performance should be quite
comparable for both implementations.

**NOTE:** Because this implementation is relatively recent, it is recommended
that for use cases where stability is paramount, the existing FIL
implementation be used.

## Usage
With one exception, experimental FIL should be fully compatible with the
existing FIL API. Experimental FIL no longer allows a `threshold` to be
specified at the time a model is loaded for binary classifiers. Instead, the
threshold must be passed as a keyword argument to the `predict` method.

Besides this, all existing FIL calls should be compatible with experimental
FIL. There are, however, several performance parameters which have been
deprecated (will now emit a warning) and a few new ones which have been added.

The most basic usage remains the same:
```python
from cuml.experimental import ForestInference

fm = ForestInference.load(filename=model_path,
                          output_class=True,
                          model_type='xgboost')

X = ... load test samples as a numpy or cupy array ...

y_out = fm.predict(X)
```

In order to optimize performance, however, we introduce a new optional
parameter to the `predict` method called `chunk_size`:

```python
y_out = fm.predict(X, chunk_size=4)
```

The API docs cover `chunk_size` in more detail, but this parameter controls
how many rows within a batch are simultaneously evaluated during a single
iteration of FIL's inference algorithm. The optimal value for this
parameter depends on both the model and available hardware, and it is
difficult to predict _a priori_. In general, however, larger batches benefit
from larger `chunk_size` values, and smaller batches benefit from smaller
`chunk_size` values.

For GPU execution, `chunk_size` can be any power of 2 from 1 to 32. For CPU
execution, `chunk_size` can be any power of 2, but there is generally no
benefit in testing values over 512. On both CPU and GPU, there is never
any benefit from a chunk size that exceeds the batch size. Tuning the
chunk size can substantially improve performance, so it is often worthwhile
to perform a search over chunk sizes with sample data when deploying a model
with FIL.

### Loading Parameters
In addition to the `chunk_size` parameter for the `predict` and
`predict_proba` methods, FIL offers some parameters for optimizing
performance when the model is loaded. This implementation also
deprecates some existing parameters.

#### Deprecated `load` Parameters

- `threshold` (will raise a `DeprecationWarning` if used)
- `algo` (ignored, but a warning will be logged)
- `storage_type` (ignored, but a warning will be logged)
- `blocks_per_sm` (ignored, but a warning will be logged)
- `threads_per_tree` (ignored, but a warning will be logged)
- `n_items` (ignored, but a warning will be logged)
- `compute_shape_str` (ignored, but a warning will be logged)

#### New `load` Parameters
- `layout`: Replaces the functionality of `algo` and specifies the in-memory
  layout of nodes in FIL forests. One of `'depth_first'` (default) or
  `'breadth_first'`. Except in cases where absolutely optimal
  performance is critical, the default should be acceptable.
- `align_bytes`: If specified, trees will be padded such that their in-memory
  size is a multiple of this value. Theoretically, this can improve
  performance by guaranteeing that memory reads from trees begin on a cache
  line boundary. Empirically, little benefit has been observed for this
  parameter, and it may be deprecated before this version of FIL moves out of
  experimental status.

#### Optimizing `load` parameters
While these two new parameters have been provided for cases in which it is
necessary to eke out every possible performance gain for a model, in general
the performance benefit will be tiny relative to the benefit of
optimizing `chunk_size` for predict calls.

## Future Development
Once experimental FIL has been thoroughly tested and evaluated in real-world
deployments, it will be moved out of experimental status and replace the
existing FIL implementation. Before this happens, RAPIDS developers will
also address the current underperformance of experimental FIL on shallow
trees to ensure performance parity.

While this version of FIL remains in experimental status, feedback is very
much welcome. Please consider [submitting an
issue](https://github.com/rapidsai/cuml/issues/new/choose) if you notice
any performance regression when transitioning from the current FIL, have
thoughts on how to make the API more useful, or have features you
would like to see in the new version of FIL before it transitions out of
experimental.
