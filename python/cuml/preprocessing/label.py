# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cudf
import cupy as cp

from cuml.prims.label import make_monotonic, check_labels, \
    invert_labels

import numba.cuda

from cuml.utils import rmm_cupy_ary


def label_binarize(y, classes, neg_label=0, pos_label=1,
                   sparse_output=False):
    """


    Parameters
    ----------
    :param classes:
    :param neg_label:
    :param pos_label:
    :param sparse_output:
    :return:
    """

    classes = cp.asarray(classes, dtype=classes.dtype)
    labels = cp.asarray(y, dtype=y.dtype)

    if not check_labels(labels, classes):
        raise ValueError("Unseen classes encountered in input")

    row_ind = cp.arange(0, labels.shape[0], 1, dtype=y.dtype)
    col_ind, _ = make_monotonic(labels, classes, copy=True)

    val = cp.full(row_ind.shape[0], pos_label, dtype=y.dtype)

    sp = cp.sparse.coo_matrix((val, (row_ind, col_ind)),
                              shape=(col_ind.shape[0],
                                     classes.shape[0]),
                              dtype=cp.float32)

    if sparse_output:
        sp = sp.tocsr()
        return sp
    else:

        arr = sp.toarray().astype(y.dtype)
        arr[arr == 0] = neg_label

        return arr


class LabelBinarizer(object):

    def __init__(self, neg_label=0, pos_label=1, sparse_output=False):
        if neg_label >= pos_label:
            raise ValueError("neg_label=%s must be less "
                             "than pos_label=%s." % (neg_label, pos_label))

        if sparse_output and (pos_label == 0 or neg_label != 0):
            raise ValueError("Sparse binarization is only supported "
                             "with non-zero"
                             "pos_label and zero neg_label, got pos_label=%s "
                             "and neg_label=%s"
                             % (pos_label, neg_label))

        self.neg_label = neg_label
        self.pos_label = pos_label
        self.sparse_output = sparse_output

    def fit(self, y):
        """
        Fit label binarizer

        Parameters
        ----------
        y : array of shape [n_samples,] or [n_samples, n_classes]
            Target values. The 2-d matrix should only contain 0 and 1,
            represents multilabel classification.

        Returns
        -------
        self : returns an instance of self.
        """

        if y.ndims > 2:
            raise ValueError("labels cannot be greater than 2 dimensions")

        if y.ndims == 2:

            unique_classes = rmm_cupy_ary(cp.unique, y)
            if unique_classes != [0, 1]:
                raise ValueError("2-d array can must be binary")

            self.classes_ = rmm_cupy_ary(cp.arange, 0, y.shape[1])
        else:
            self.classes_ = rmm_cupy_ary(cp.unique, y).astype(y.dtype)

        return self

    def fit_transform(self, y):
        """
        Fit label binarizer and transform multi-class labels to binary

        Parameters
        ----------
        y : array of shape [n_samples,] or [n_samples, n_classes]
        :param y:
        :return:
        """
        return self.fit(y).transform(y)

    def transform(self, y):
        """

        labels.
        :param y:
        :return:
        """
        return label_binarize(y, self.classes_,
                              pos_label=self.pos_label,
                              neg_label=self.neg_label,
                              sparse_output=self.sparse_output)

    def inverse_transform(self, y, threshold=None):
        """
        Transform binary labels back to multi-class labels
        :param Y:
        :param threshold:
        :return:
        """

        # If we are already given multi-class, just return it.
        if cp.sparse.isspmatrix(y):
            y_mapped = y.tocsr().indices.astype(self.classes_.dtype)
        else:
            y_mapped = cp.argmax(
                cp.asarray(y, dtype=y.dtype), axis=1).astype(y.dtype)

        return invert_labels(y_mapped, self.classes_)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['classes_'] = cudf.Series(numba.cuda.to_device(self.classes_))
        return state

    def __setstate__(self, state):
        state['classes_'] = cp.asarray(state["classes_"].to_gpu_array(),
                                       dtype=self.classes_.dtype)
        self.__dict__.update(state)
