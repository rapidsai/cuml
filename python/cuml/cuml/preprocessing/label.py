# Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

from cuml.prims.label import check_labels, invert_labels, make_monotonic
from cuml.internals.array_sparse import SparseCumlArray
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common import CumlArray, has_scipy
import cuml.internals
from cuml import Base
from cuml.internals.safe_imports import gpu_only_import

cp = gpu_only_import("cupy")
cupyx = gpu_only_import("cupyx")


@cuml.internals.api_return_sparse_array()
def label_binarize(
    y, classes, neg_label=0, pos_label=1, sparse_output=False
) -> SparseCumlArray:
    """
    A stateless helper function to dummy encode multi-class labels.

    Parameters
    ----------

    y : array-like of size [n_samples,] or [n_samples, n_classes]
    classes : the set of unique classes in the input
    neg_label : integer the negative value for transformed output
    pos_label : integer the positive value for transformed output
    sparse_output : bool whether to return sparse array
    """

    classes = cp.asarray(classes, dtype=classes.dtype)
    labels = cp.asarray(y, dtype=y.dtype)

    if not check_labels(labels, classes):
        raise ValueError("Unseen classes encountered in input")

    row_ind = cp.arange(0, labels.shape[0], 1, dtype=y.dtype)
    col_ind, _ = make_monotonic(labels, classes, copy=True)

    # Convert from CumlArray to cupy
    col_ind = cp.asarray(col_ind)

    val = cp.full(row_ind.shape[0], pos_label, dtype=y.dtype)

    sp = cupyx.scipy.sparse.coo_matrix(
        (val, (row_ind, col_ind)),
        shape=(col_ind.shape[0], classes.shape[0]),
        dtype=cp.float32,
    )

    cp.cuda.Stream.null.synchronize()

    is_binary = classes.shape[0] == 2

    if sparse_output:
        sp = sp.tocsr()
        if is_binary:
            sp = sp.getcol(1)  # getcol does not support -1 indexing
        return sp
    else:

        arr = sp.toarray().astype(y.dtype)
        arr[arr == 0] = neg_label
        if is_binary:
            arr = arr[:, -1].reshape((-1, 1))
        return arr


class LabelBinarizer(Base):

    """
    A multi-class dummy encoder for labels.

    Parameters
    ----------

    neg_label : integer (default=0)
        label to be used as the negative binary label
    pos_label : integer (default=1)
        label to be used as the positive binary label
    sparse_output : bool (default=False)
        whether to return sparse arrays for transformed output
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Examples
    --------

    Create an array with labels and dummy encode them

    .. code-block:: python

        >>> import cupy as cp
        >>> import cupyx
        >>> from cuml.preprocessing import LabelBinarizer

        >>> labels = cp.asarray([0, 5, 10, 7, 2, 4, 1, 0, 0, 4, 3, 2, 1],
        ...                     dtype=cp.int32)

        >>> lb = LabelBinarizer()
        >>> encoded = lb.fit_transform(labels)
        >>> print(str(encoded))
        [[1 0 0 0 0 0 0 0]
        [0 0 0 0 0 1 0 0]
        [0 0 0 0 0 0 0 1]
        [0 0 0 0 0 0 1 0]
        [0 0 1 0 0 0 0 0]
        [0 0 0 0 1 0 0 0]
        [0 1 0 0 0 0 0 0]
        [1 0 0 0 0 0 0 0]
        [1 0 0 0 0 0 0 0]
        [0 0 0 0 1 0 0 0]
        [0 0 0 1 0 0 0 0]
        [0 0 1 0 0 0 0 0]
        [0 1 0 0 0 0 0 0]]
        >>> decoded = lb.inverse_transform(encoded)
        >>> print(str(decoded))
        [ 0  5 10  7  2  4  1  0  0  4  3  2  1]

    """

    classes_ = CumlArrayDescriptor()

    def __init__(
        self,
        *,
        neg_label=0,
        pos_label=1,
        sparse_output=False,
        handle=None,
        verbose=False,
        output_type=None,
    ):
        super().__init__(
            handle=handle, verbose=verbose, output_type=output_type
        )

        if neg_label >= pos_label:
            raise ValueError(
                "neg_label=%s must be less "
                "than pos_label=%s." % (neg_label, pos_label)
            )

        if sparse_output and (pos_label == 0 or neg_label != 0):
            raise ValueError(
                "Sparse binarization is only supported "
                "with non-zero"
                "pos_label and zero neg_label, got pos_label=%s "
                "and neg_label=%s" % (pos_label, neg_label)
            )

        self.neg_label = neg_label
        self.pos_label = pos_label
        self.sparse_output = sparse_output
        self.classes_ = None

    def fit(self, y) -> "LabelBinarizer":
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

        if y.ndim > 2:
            raise ValueError("labels cannot be greater than 2 dimensions")

        if y.ndim == 2:

            unique_classes = cp.unique(y)
            if unique_classes != [0, 1]:
                raise ValueError("2-d array can must be binary")

            self.classes_ = cp.arange(0, y.shape[1])
        else:
            self.classes_ = cp.unique(y).astype(y.dtype)

        cp.cuda.Stream.null.synchronize()

        return self

    def fit_transform(self, y) -> SparseCumlArray:
        """
        Fit label binarizer and transform multi-class labels to their
        dummy-encoded representation.

        Parameters
        ----------
        y : array of shape [n_samples,] or [n_samples, n_classes]

        Returns
        -------

        arr : array with encoded labels
        """
        return self.fit(y).transform(y)

    def transform(self, y) -> SparseCumlArray:
        """
        Transform multi-class labels to their dummy-encoded representation
        labels.

        Parameters
        ----------
        y : array of shape [n_samples,] or [n_samples, n_classes]

        Returns
        -------
        arr : array with encoded labels
        """
        return label_binarize(
            y,
            self.classes_,
            pos_label=self.pos_label,
            neg_label=self.neg_label,
            sparse_output=self.sparse_output,
        )

    def inverse_transform(self, y, threshold=None) -> CumlArray:
        """
        Transform binary labels back to original multi-class labels

        Parameters
        ----------

        y : array of shape [n_samples, n_classes]
        threshold : float this value is currently ignored

        Returns
        -------

        arr : array with original labels
        """

        if has_scipy():
            from scipy.sparse import isspmatrix as scipy_sparse_isspmatrix
        else:
            from cuml.internals.import_utils import (
                dummy_function_always_false as scipy_sparse_isspmatrix,
            )

        # If we are already given multi-class, just return it.
        if cupyx.scipy.sparse.isspmatrix(y):
            y_mapped = y.tocsr().indices.astype(self.classes_.dtype)
        elif scipy_sparse_isspmatrix(y):
            y = y.tocsr()
            y_mapped = cp.array(y.indices, dtype=y.indices.dtype)
        else:
            y_mapped = cp.argmax(cp.asarray(y, dtype=y.dtype), axis=1).astype(
                y.dtype
            )

        return invert_labels(y_mapped, self.classes_)

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + [
            "neg_label",
            "pos_label",
            "sparse_output",
        ]
