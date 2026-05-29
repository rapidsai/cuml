# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import cupy as cp
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import ExtraTreesClassifier as sketc
from sklearn.ensemble import ExtraTreesRegressor as sketr
from sklearn.metrics import accuracy_score

from cuml.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from cuml.metrics import r2_score
from cuml.testing.utils import quality_param, stress_param, unit_param


@pytest.fixture(scope="module")
def small_clf():
    X, y = make_classification(
        n_samples=400, n_features=8, n_informative=4, n_redundant=2,
        random_state=42,
    )
    return X.astype(np.float32), y.astype(np.int32)


@pytest.fixture(scope="module")
def small_reg():
    X, y = make_regression(
        n_samples=400, n_features=8, n_informative=4, random_state=42,
    )
    return X.astype(np.float32), y.astype(np.float32)


@pytest.fixture(scope="module")
def imbalanced_clf():
    X, y = make_classification(
        n_samples=400, n_features=8, n_informative=4, n_redundant=2,
        weights=[0.9, 0.1], random_state=42,
    )
    return X.astype(np.float32), y.astype(np.int32)


def test_default_parameters_classifier():
    clf = ExtraTreesClassifier()
    assert clf.bootstrap is False
    assert clf._splitter == "random"
    assert clf._cpu_class_path == "sklearn.ensemble.ExtraTreesClassifier"


def test_default_parameters_regressor():
    reg = ExtraTreesRegressor()
    assert reg.bootstrap is False
    assert reg._splitter == "random"
    assert reg._cpu_class_path == "sklearn.ensemble.ExtraTreesRegressor"


@pytest.mark.parametrize(
    "estimator_cls", [ExtraTreesClassifier, ExtraTreesRegressor]
)
def test_clone_preserves_splitter(small_clf, small_reg, estimator_cls):
    from sklearn.base import clone

    X, y = (
        small_clf if estimator_cls is ExtraTreesClassifier else small_reg
    )
    est = estimator_cls(
        n_estimators=5, max_depth=4, random_state=42, n_streams=1,
    )
    cloned = clone(est)
    assert isinstance(cloned, estimator_cls)
    assert cloned._splitter == "random"
    cloned.fit(X, y)
    assert cloned._splitter == "random"


def test_unknown_splitter_raises(small_clf):
    X, y = small_clf

    class _BadSplitter(ExtraTreesClassifier):
        _splitter = "bogus"

    est = _BadSplitter(n_estimators=2, max_depth=2, n_streams=1)
    with pytest.raises(
        ValueError, match=r"Unknown _splitter=.*bogus.*expected one of",
    ):
        est.fit(X, y)


def test_extra_trees_classification(small_clf):
    X, y = small_clf

    cu_clf = ExtraTreesClassifier(
        n_estimators=50, max_depth=8, max_features="sqrt", random_state=42,
        n_streams=1,
    ).fit(X, y)
    sk_clf = sketc(
        n_estimators=50, max_depth=8, max_features="sqrt", random_state=42,
    ).fit(X, y)

    cu_acc = accuracy_score(y, cu_clf.predict(X))
    sk_acc = accuracy_score(y, sk_clf.predict(X))

    # 10-seed sweep on this fixture: mean=0.010, range=[-0.020, 0.035],
    # stderr=0.004 for (sk_acc - cu_acc). 0.07 leaves comfortable headroom.
    assert cu_acc >= (sk_acc - 0.07)


def test_extra_trees_regression(small_reg):
    X, y = small_reg

    cu_reg = ExtraTreesRegressor(
        n_estimators=50, max_depth=8, max_features=1.0, random_state=42,
        n_streams=1,
    ).fit(X, y)
    sk_reg = sketr(
        n_estimators=50, max_depth=8, max_features=1.0, random_state=42,
    ).fit(X, y)

    cu_r2 = r2_score(y, cu_reg.predict(X))
    sk_r2 = sk_reg.score(X, y)

    # 10-seed sweep on this fixture: mean=-0.005, range=[-0.010, 0.001],
    # stderr=0.001 for (sk_r2 - cu_r2). cuml is on-par or slightly ahead
    # of sklearn here; the 0.10 tolerance is for cross-platform headroom.
    assert cu_r2 >= (sk_r2 - 0.10)


def test_sample_weight_shifts_classifier_predictions(small_clf):
    X, y = small_clf

    # Up-weight class-1 rows by 5x, leaving class-0 at unit weight.
    sample_weight = np.where(y == 1, 5.0, 1.0).astype(np.float32)

    clf_unweighted = ExtraTreesClassifier(
        n_estimators=30, max_depth=6, max_features="sqrt", random_state=42,
        n_streams=1,
    ).fit(X, y)
    clf_weighted = ExtraTreesClassifier(
        n_estimators=30, max_depth=6, max_features="sqrt", random_state=42,
        n_streams=1,
    ).fit(X, y, sample_weight=sample_weight)

    p_unweighted = clf_unweighted.predict(X)
    p_weighted = clf_weighted.predict(X)

    # The weighted fit should disagree with the unweighted fit on enough rows
    # to demonstrate sample_weight is reaching the splitter (not silently
    # ignored). Observed: ~7-15% of rows flip across seeds.
    flipped = (p_unweighted != p_weighted).mean()
    assert flipped > 0.01, f"sample_weight had no effect (flipped={flipped})"


def test_sample_weight_shifts_regressor_predictions(small_reg):
    X, y = small_reg

    # Up-weight the upper half of y by 5x.
    sample_weight = np.where(y > np.median(y), 5.0, 1.0).astype(np.float32)

    reg_unweighted = ExtraTreesRegressor(
        n_estimators=30, max_depth=6, max_features=1.0, random_state=42,
        n_streams=1,
    ).fit(X, y)
    reg_weighted = ExtraTreesRegressor(
        n_estimators=30, max_depth=6, max_features=1.0, random_state=42,
        n_streams=1,
    ).fit(X, y, sample_weight=sample_weight)

    p_unweighted = reg_unweighted.predict(X)
    p_weighted = reg_weighted.predict(X)

    # Mean prediction should shift upward when upper-half rows carry 5x
    # weight; magnitude varies with random splits but is reliably positive.
    mean_shift = np.asarray(p_weighted).mean() - np.asarray(p_unweighted).mean()
    assert mean_shift > 0.0, (
        f"upweighting upper-half y did not shift predictions up: "
        f"shift={mean_shift}"
    )


@pytest.mark.parametrize("class_weight", ["balanced", {0: 1, 1: 5}])
def test_class_weight_recovers_minority(imbalanced_clf, class_weight):
    X, y = imbalanced_clf

    clf_plain = ExtraTreesClassifier(
        n_estimators=30, max_depth=6, random_state=42, n_streams=1,
    ).fit(X, y)
    clf_weighted = ExtraTreesClassifier(
        n_estimators=30, max_depth=6, random_state=42, n_streams=1,
        class_weight=class_weight,
    ).fit(X, y)

    minority = (y == 1)
    # Unweighted recall on the minority class tends to drift below 0.6 on
    # imbalanced fixtures; class_weight should lift it.
    recall_plain = (clf_plain.predict(X)[minority] == 1).mean()
    recall_weighted = (clf_weighted.predict(X)[minority] == 1).mean()
    assert recall_weighted >= recall_plain


def test_class_weight_balanced_subsample_default_bootstrap_false_collapses(
    small_clf,
):
    # ExtraTreesClassifier defaults bootstrap=False; balanced_subsample
    # collapses to 'balanced' with a UserWarning and the resulting
    # predictions must match an explicit 'balanced' fit byte-for-byte.
    X, y = small_clf
    with pytest.warns(
        UserWarning,
        match=r"balanced_subsample.*requires bootstrap=True.*falling back to 'balanced'",
    ):
        bs = ExtraTreesClassifier(
            n_estimators=20,
            max_depth=6,
            n_streams=1,
            random_state=42,
            class_weight="balanced_subsample",
        ).fit(X, y)
    b = ExtraTreesClassifier(
        n_estimators=20,
        max_depth=6,
        n_streams=1,
        random_state=42,
        class_weight="balanced",
    ).fit(X, y)
    cp.testing.assert_array_equal(bs.predict(X), b.predict(X))


def test_class_weight_balanced_subsample_bootstrap_true_lifts_minority_recall():
    # On bootstrap=True (so the per-tree compute path actually runs), the
    # mode lifts minority-class recall vs the unweighted ET on an
    # imbalanced 90:10 fixture; the bootstrap=False UserWarning must not
    # fire on this path.
    import warnings

    from sklearn.metrics import recall_score

    rng = np.random.default_rng(0)
    X_maj = rng.normal(0.0, 1.0, size=(360, 4)).astype(np.float32)
    X_min = rng.normal(1.5, 1.0, size=(40, 4)).astype(np.float32)
    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(360), np.ones(40)]).astype(np.int32)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message=r"balanced_subsample.*falling back",
            category=UserWarning,
        )
        bs = ExtraTreesClassifier(
            n_estimators=50,
            max_depth=8,
            n_streams=1,
            random_state=42,
            bootstrap=True,
            class_weight="balanced_subsample",
        ).fit(X, y)
    unw = ExtraTreesClassifier(
        n_estimators=50,
        max_depth=8,
        n_streams=1,
        random_state=42,
        bootstrap=True,
    ).fit(X, y)
    assert recall_score(y, bs.predict(X), pos_label=1) > recall_score(
        y, unw.predict(X), pos_label=1
    )


def test_bootstrap_true_oob_score(small_clf):
    X, y = small_clf
    clf = ExtraTreesClassifier(
        n_estimators=30, max_depth=6, random_state=42, n_streams=1,
        bootstrap=True, oob_score=True,
    ).fit(X, y)
    assert hasattr(clf, "oob_score_")
    assert 0.0 < clf.oob_score_ <= 1.0


def test_n_bins_too_small_raises(small_clf):
    X, y = small_clf
    clf = ExtraTreesClassifier(
        n_estimators=5, n_bins=1, max_depth=4, random_state=42,
    )
    # The SPLITTER_RANDOM path requires n_bins >= 2 (the C++ et_split_position
    # Lemire chain divides by n_bins - 1).
    with pytest.raises(RuntimeError, match=r"SPLITTER_RANDOM requires max_n_bins"):
        clf.fit(X, y)


@pytest.mark.parametrize(
    "scale",
    [
        unit_param({"n_samples": 1000, "n_features": 20}),
        quality_param({"n_samples": 10000, "n_features": 100}),
        stress_param({"n_samples": 200000, "n_features": 200}),
    ],
)
def test_extra_trees_scaling(scale):
    X, y = make_classification(
        n_samples=scale["n_samples"],
        n_features=scale["n_features"],
        n_informative=max(5, scale["n_features"] // 4),
        random_state=42,
    )
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    clf = ExtraTreesClassifier(
        n_estimators=20, max_depth=8, max_features="sqrt", random_state=42,
        n_streams=1,
    ).fit(X, y)
    assert accuracy_score(y, clf.predict(X)) > 0.5
