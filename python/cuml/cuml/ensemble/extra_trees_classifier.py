# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from cuml.ensemble.randomforestclassifier import RandomForestClassifier


class ExtraTreesClassifier(RandomForestClassifier):
    """
    Implements an Extra-Trees (extremely randomized trees) classifier model.
    Each tree node draws a single random threshold per candidate feature
    rather than searching for the optimal split, which trains faster than a
    regular Random Forest and produces higher-variance individual trees.
    Defaults to ``bootstrap=False``, matching scikit-learn's
    ``ExtraTreesClassifier``.

    .. note:: The underlying algorithm for tree node splits differs
      from that used in scikit-learn. cuML draws the random threshold from
      one of `n_bins` quantile positions per feature; you can tune the
      precision with the ``n_bins`` parameter.

    Examples
    --------

    .. code-block:: python

        >>> import cupy as cp
        >>> from cuml.ensemble import ExtraTreesClassifier as cuETC

        >>> X = cp.random.normal(size=(10, 4)).astype(cp.float32)
        >>> y = cp.asarray([0, 1] * 5, dtype=cp.int32)

        >>> cuml_model = cuETC(max_features=1.0, n_bins=8, n_estimators=40)
        >>> cuml_model.fit(X, y)
        ExtraTreesClassifier()
        >>> cuml_predict = cuml_model.predict(X)

    Parameters
    ----------
    bootstrap : boolean, default=False
        Control bootstrapping. Defaults to ``False`` (the sklearn ExtraTrees
        default), unlike ``RandomForestClassifier`` which defaults to ``True``.

    All other parameters are inherited from
    :class:`~cuml.ensemble.RandomForestClassifier`; see that class for the
    full parameter and attribute documentation.

    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Notes
    -----
    For additional docs, see `scikit-learn's ExtraTreesClassifier
    <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html>`_.
    """

    _splitter = "random"
    _cpu_class_path = "sklearn.ensemble.ExtraTreesClassifier"

    def __init__(
        self,
        *,
        n_estimators=100,
        split_criterion="gini",
        bootstrap=False,
        max_samples=1.0,
        max_depth="deprecated",
        max_leaves=-1,
        max_features="sqrt",
        n_bins=128,
        min_samples_leaf=1,
        min_samples_split=2,
        min_impurity_decrease=0.0,
        max_batch_size=4096,
        random_state=None,
        n_streams=4,
        oob_score=False,
        class_weight=None,
        verbose=False,
        output_type=None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            split_criterion=split_criterion,
            bootstrap=bootstrap,
            max_samples=max_samples,
            max_depth=max_depth,
            max_leaves=max_leaves,
            max_features=max_features,
            n_bins=n_bins,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            max_batch_size=max_batch_size,
            random_state=random_state,
            n_streams=n_streams,
            oob_score=oob_score,
            class_weight=class_weight,
            verbose=verbose,
            output_type=output_type,
        )
