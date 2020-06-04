#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.exceptions import NotFittedError
import cupy as cp
from cuml.common import with_cupy_rmm
from cuml.common.sparsefuncs import csr_row_normalize_l1
from cuml.common.sparsefuncs import csr_row_normalize_l2


def _sparse_document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if cp.sparse.isspmatrix_csr(X):
        return cp.bincount(X.indices, minlength=X.shape[1])
    else:
        return cp.diff(X.indptr)


class TfidfTransformer:
    """Transform a count matrix to a normalized tf or tf-idf representation
    Tf means term-frequency while tf-idf means term-frequency times inverse
    document-frequency. This is a common term weighting scheme in information
    retrieval, that has also found good use in document classification.
    The goal of using tf-idf instead of the raw frequencies of occurrence of a
    token in a given document is to scale down the impact of tokens that occur
    very frequently in a given corpus and that are hence empirically less
    informative than features that occur in a small fraction of the training
    corpus.
    The formula that is used to compute the tf-idf for a term t of a document d
    in a document set is tf-idf(t, d) = tf(t, d) * idf(t), and the idf is
    computed as idf(t) = log [ n / df(t) ] + 1 (if ``smooth_idf=False``), where
    n is the total number of documents in the document set and df(t) is the
    document frequency of t; the document frequency is the number of documents
    in the document set that contain the term t. The effect of adding "1" to
    the idf in the equation above is that terms with zero idf, i.e., terms
    that occur in all documents in a training set, will not be entirely
    ignored.
    (Note that the idf formula above differs from the standard textbook
    notation that defines the idf as
    idf(t) = log [ n / (df(t) + 1) ]).
    If ``smooth_idf=True`` (the default), the constant "1" is added to the
    numerator and denominator of the idf as if an extra document was seen
    containing every term in the collection exactly once, which prevents
    zero divisions: idf(t) = log [ (1 + n) / (1 + df(t)) ] + 1.
    Furthermore, the formulas used to compute tf and idf depend
    on parameter settings that correspond to the SMART notation used in IR
    as follows:
    Tf is "n" (natural) by default, "l" (logarithmic) when
    ``sublinear_tf=True``.
    Idf is "t" when use_idf is given, "n" (none) otherwise.
    Normalization is "c" (cosine) when ``norm='l2'``, "n" (none)
    when ``norm=None``.

    Parameters
    ----------
    norm : {'l1', 'l2'}, default='l2'
        Each output row will have unit norm, either:
        * 'l2': Sum of squares of vector elements is 1. The cosine
        similarity between two vectors is their dot product when l2 norm has
        been applied.
        * 'l1': Sum of absolute values of vector elements is 1.
    use_idf : bool, default=True
        Enable inverse-document-frequency reweighting.
    smooth_idf : bool, default=True
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.
    sublinear_tf : bool, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    Attributes
    ----------
    idf_ : array of shape (n_features)
        The inverse document frequency (IDF) vector; only defined
        if  ``use_idf`` is True.
    """

    def __init__(self, *, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    @with_cupy_rmm
    def fit(self, X):
        """Learn the idf vector (global term weights).

        Parameters
        ----------
        X : array-like of shape n_samples, n_features
            A matrix of term/token counts.
        """
        dtype = X.dtype if X.dtype in FLOAT_DTYPES else cp.float32
        X = self._convert_to_csr(X, dtype)

        if self.use_idf:
            n_samples, n_features = X.shape
            df = _sparse_document_frequency(X)
            df = df.astype(dtype, copy=False)

            # perform idf smoothing if required
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            # log+1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            idf = cp.log(n_samples / df) + 1
            self._idf_diag = cp.sparse.dia_matrix(
                (idf, 0),
                shape=(n_features, n_features),
                dtype=dtype
            )

        return self

    @with_cupy_rmm
    def transform(self, X, copy=True):
        """Transform a count matrix to a tf or tf-idf representation

        Parameters
        ----------
        X : array-like of (n_samples, n_features)
            A matrix of term/token counts
        copy : bool, default=True
            Whether to copy X and operate on the copy or perform in-place
            operations.

        Returns
        -------
        vectors : array-like of shape (n_samples, n_features)
        """
        if copy:
            X = X.copy()

        dtype = X.dtype if X.dtype in FLOAT_DTYPES else cp.float32

        X = self._convert_to_csr(X, dtype)
        if X.dtype != dtype:
            X = X.astype(dtype)

        n_samples, n_features = X.shape

        if self.sublinear_tf:
            cp.log(X.data, X.data)
            X.data += 1

        if self.use_idf:
            self._check_is_idf_fitted()

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            X *= self._idf_diag

        if self.norm:
            if self.norm == 'l1':
                csr_row_normalize_l1(X, inplace=True)
            elif self.norm == 'l2':
                csr_row_normalize_l2(X, inplace=True)

        return X

    def fit_transform(self, X, copy=True):
        """
        Fit TfidfTransformer to X, then transform X.
        Equivalent to fit(X).transform(X).

        Parameters
        ----------
        X : array-like of (n_samples, n_features)
            A matrix of term/token counts
        copy : bool, default=True
            Whether to copy X and operate on the copy or perform in-place
            operations.
        Returns
        -------
        vectors : array-like of shape (n_samples, n_features)
        """
        return self.fit(X).transform(X, copy=copy)

    def _check_is_idf_fitted(self):
        if not hasattr(self, 'idf_'):
            msg = ("This TfidfTransformer instance is not fitted or the "
                   "value of use_idf is not consistant between "
                   ".fit() and .transform().")
            raise NotFittedError(msg)

    def _convert_to_csr(self, X, dtype):
        """Convert array to CSR format if it not sparse nor CSR."""
        if not cp.sparse.isspmatrix_csr(X):
            if not cp.sparse.issparse(X):
                X = cp.sparse.csr_matrix(X.astype(dtype))
            else:
                X = X.tocsr()
        return X

    @property
    def idf_(self):
        # if _idf_diag is not set, this will raise an attribute error,
        # which means hasattr(self, "idf_") is False
        return self._idf_diag.data

    @idf_.setter
    def idf_(self, value):
        value = cp.asarray(value, dtype=cp.float32)
        n_features = value.shape[0]
        self._idf_diag = cp.sparse.dia_matrix(
            (value, 0),
            shape=(n_features, n_features),
            dtype=cp.float32
        )
