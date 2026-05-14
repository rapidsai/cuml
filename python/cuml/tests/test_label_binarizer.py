# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pytest
import scipy.sparse as sp

from cuml.preprocessing import LabelBinarizer, label_binarize


def test_label_binarizer_no_features():
    """Ensure the features infra is never applied to LabelBinarizer"""
    y = cp.asarray([1, 2, 1, 2, 1, 0])
    model = LabelBinarizer().fit(y)
    assert not hasattr(model, "n_features_in_")


def test_label_binarizer_not_fitted():
    lb = LabelBinarizer()
    err_msg = "This LabelBinarizer instance is not fitted yet"
    with pytest.raises(ValueError, match=err_msg):
        lb.transform([])
    with pytest.raises(ValueError, match=err_msg):
        lb.inverse_transform([])


def test_label_binarizer_invalid_parameters():
    input_labels = [0, 1, 0, 1]
    err_msg = "neg_label=2 must be strictly less than pos_label=1."
    lb = LabelBinarizer(neg_label=2, pos_label=1)
    with pytest.raises(ValueError, match=err_msg):
        lb.fit(input_labels)
    err_msg = "neg_label=2 must be strictly less than pos_label=2."
    lb = LabelBinarizer(neg_label=2, pos_label=2)
    with pytest.raises(ValueError, match=err_msg):
        lb.fit(input_labels)
    err_msg = (
        "Sparse binarization is only supported with non zero pos_label and zero "
        "neg_label, got pos_label=2 and neg_label=1"
    )
    lb = LabelBinarizer(neg_label=1, pos_label=2, sparse_output=True)
    with pytest.raises(ValueError, match=err_msg):
        lb.fit(input_labels)


def test_label_binarizer_invalid_y_types():
    err_msg = (
        "Multioutput target data is not supported with label binarization"
    )
    with pytest.raises(ValueError, match=err_msg):
        LabelBinarizer().fit(np.array([[1, 3], [2, 1]]))
    with pytest.raises(ValueError, match=err_msg):
        label_binarize(np.array([[1, 3], [2, 1]]), classes=[1, 2, 3])

    with pytest.raises(ValueError, match="Unknown label type: continuous"):
        LabelBinarizer().fit([1.2, 2.7])


def test_label_binarize_mismatch_labels():
    with pytest.raises(ValueError, match="mismatch with the labels"):
        label_binarize([[1, 0]], classes=[0, 1, 2])

    lb = LabelBinarizer().fit(np.array([[0, 0, 1]]))
    with pytest.raises(ValueError, match="mismatch with the labels"):
        lb.transform(np.array([[0, 1]]))


def toarray(x, expect_sparse):
    if expect_sparse:
        assert sp.issparse(x)
        return x.toarray()
    return x


@pytest.mark.parametrize(
    "sparse_output, p, n",
    [
        (False, 1, 0),
        (False, 10, 0),
        (False, 0, -10),
        (False, 1, -1),
        (True, 1, 0),
        (True, 10, 0),
    ],
)
def test_label_binarizer_one_class(sparse_output, p, n):
    y = np.array(["a", "a", "a"])
    lb = LabelBinarizer(sparse_output=sparse_output, pos_label=p, neg_label=n)

    # fit_transform
    sol = np.array([n, n, n])[:, None]
    res = toarray(lb.fit_transform(y), sparse_output)
    np.testing.assert_array_equal(res, sol)

    # transform
    res = toarray(lb.transform(y), sparse_output)
    np.testing.assert_array_equal(res, sol)

    # attributes
    assert lb.y_type_ == "binary"
    assert not lb.sparse_input_
    np.testing.assert_array_equal(lb.classes_, np.array(["a"]))

    # inverse_transform
    np.testing.assert_array_equal(lb.inverse_transform(res), y)

    # transform unseen labels
    unseen = np.array(["a", "b"])
    res = toarray(lb.transform(unseen), sparse_output)
    sol = np.array([n, n])[:, None]
    np.testing.assert_array_equal(res, sol)


@pytest.mark.parametrize(
    "sparse_output, p, n",
    [
        (False, 1, 0),
        (False, 10, 0),
        (False, 0, -10),
        (False, 1, -1),
        (True, 1, 0),
        (True, 10, 0),
    ],
)
def test_label_binarizer_two_classes(sparse_output, p, n):
    y = np.array(["a", "b", "b", "a"])
    lb = LabelBinarizer(sparse_output=sparse_output, pos_label=p, neg_label=n)

    # fit_transform
    sol = np.array([n, p, p, n])[:, None]
    res = toarray(lb.fit_transform(y), sparse_output)
    np.testing.assert_array_equal(res, sol)

    # transform
    res = toarray(lb.transform(y), sparse_output)
    np.testing.assert_array_equal(res, sol)

    # attributes
    assert lb.y_type_ == "binary"
    assert not lb.sparse_input_
    np.testing.assert_array_equal(lb.classes_, np.array(["a", "b"]))

    # inverse_transform
    np.testing.assert_array_equal(lb.inverse_transform(res), y)
    # can also invert 2 column output
    y2 = np.array([[p, n], [n, p], [n, p], [p, n]])
    np.testing.assert_array_equal(lb.inverse_transform(y2), y)

    # transform of unseen classes results in 2 columns
    unseen = np.array(["a", "b", "c"])
    res = toarray(lb.transform(unseen), sparse_output)
    sol = np.array([[p, n], [n, p], [n, n]])
    np.testing.assert_array_equal(res, sol)


@pytest.mark.parametrize(
    "sparse_output, p, n",
    [
        (False, 1, 0),
        (False, 10, 0),
        (False, 0, -10),
        (False, 1, -1),
        (True, 1, 0),
        (True, 10, 0),
    ],
)
def test_label_binarizer_multiclass(sparse_output, p, n):
    y = np.array(["a", "b", "b", "a", "c"])
    lb = LabelBinarizer(sparse_output=sparse_output, pos_label=p, neg_label=n)

    # fit_transform
    sol = np.array([[p, n, n], [n, p, n], [n, p, n], [p, n, n], [n, n, p]])
    res = toarray(lb.fit_transform(y), sparse_output)
    np.testing.assert_array_equal(res, sol)

    # transform
    res = toarray(lb.transform(y), sparse_output)
    np.testing.assert_array_equal(res, sol)

    # attributes
    assert lb.y_type_ == "multiclass"
    assert not lb.sparse_input_
    np.testing.assert_array_equal(lb.classes_, np.array(["a", "b", "c"]))

    # inverse_transform
    np.testing.assert_array_equal(lb.inverse_transform(res), y)

    # transform of unseen inputs results in all 0s
    unseen = np.array(["d", "a", "e", "c", "f"])
    res = toarray(lb.transform(unseen), sparse_output)
    sol = np.array([[n, n, n], [p, n, n], [n, n, n], [n, n, p], [n, n, n]])
    np.testing.assert_array_equal(res, sol)


@pytest.mark.parametrize(
    "sparse_output, p, n",
    [
        (False, 1, 0),
        (False, 10, 0),
        (False, 0, -10),
        (False, 1, -1),
        (True, 1, 0),
        (True, 10, 0),
    ],
)
@pytest.mark.parametrize("sparse_input", [False, True])
def test_label_binarizer_multilabel_indicator(
    sparse_input, sparse_output, p, n
):
    y = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    if sparse_input:
        y = sp.csr_matrix(y)
    lb = LabelBinarizer(sparse_output=sparse_output, pos_label=p, neg_label=n)

    # fit_transform
    sol = np.array([[n, n, p], [p, n, n], [n, p, n]])
    res = toarray(lb.fit_transform(y), sparse_output)
    np.testing.assert_array_equal(res, sol)

    # transform
    res = toarray(lb.transform(y), sparse_output)
    np.testing.assert_array_equal(res, sol)

    # attributes
    assert lb.y_type_ == "multilabel-indicator"
    assert lb.sparse_input_ is sparse_input
    np.testing.assert_array_equal(lb.classes_, np.array([0, 1, 2]))

    # inverse_transform accepts both sparse and dense,
    # and returns the type used for fit
    sol = np.array([[0, 1, 0], [1, 0, 0]])
    y_dense = np.array([[n, p, n], [p, n, n]])
    y_sparse = sp.csr_matrix(y_dense)
    res = toarray(lb.inverse_transform(y_dense), sparse_input)
    np.testing.assert_array_equal(res, sol)
    res = toarray(lb.inverse_transform(y_sparse), sparse_input)
    np.testing.assert_array_equal(res, sol)

    # Can also transform non-multilabel input
    y = np.array([3, 2, 1, 0])
    sol = np.array([[n, n, n], [n, n, p], [n, p, n], [p, n, n]])
    res = toarray(lb.transform(y), sparse_output)
    np.testing.assert_array_equal(res, sol)


@pytest.mark.parametrize("threshold", [1, -1])
@pytest.mark.parametrize("sparse", [False, True])
def test_label_binarizer_inverse_transform_threshold(threshold, sparse):
    lb = LabelBinarizer().fit(np.array(["a", "b", "c"]))

    p = threshold + 1
    n = threshold - 1
    y = np.array([[n, p, n], [p, n, n], [n, n, p]])
    sol = np.array(["b", "a", "c"])
    if sparse:
        y = sp.csr_matrix(y)

    res = lb.inverse_transform(y, threshold=threshold)
    np.testing.assert_array_equal(res, sol)


def test_label_binarize_respects_class_order():
    out = label_binarize([1, 6], classes=[1, 2, 4, 6])
    expected = cp.array([[1, 0, 0, 0], [0, 0, 0, 1]])
    cp.testing.assert_array_equal(out, expected)

    # Modified class order
    out = label_binarize([1, 6], classes=[1, 6, 4, 2])
    expected = cp.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    cp.testing.assert_array_equal(out, expected)

    out = label_binarize([0, 1, 2, 3], classes=[3, 2, 0, 1])
    expected = cp.array(
        [[0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]]
    )
    cp.testing.assert_array_equal(out, expected)
