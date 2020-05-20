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
from cuml.common.input_utils import input_to_cuml_array
from cuml.common import with_cupy_rmm


# TODO: Move this to feature_extraction/text.py once CountVectorizer is merged

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
        X, _, _, _ = input_to_cuml_array(X)

        dtype = X.dtype if X.dtype in FLOAT_DTYPES else cp.float64

        if self.use_idf:
            n_samples, n_features = X.shape
            df = cp.count_nonzero(X, axis=0)
            df = df.astype(dtype)

            # perform idf smoothing if required
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            # log+1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            idf = cp.log(n_samples / df) + 1
            self.idf_ = idf

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
        dtype = X.dtype if X.dtype in FLOAT_DTYPES else cp.float64
        X, _, _, _ = input_to_cuml_array(X, deepcopy=copy,
                                         convert_to_dtype=dtype)

        n_samples, n_features = X.shape

        if self.sublinear_tf:
            idx = cp.nonzero(cp.not_equal(X, 0))
            X[idx] = cp.log(X[idx]) + 1

        if self.use_idf:
            self._check_is_idf_fitted()

            expected_n_features = len(self.idf_)
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            X = cp.multiply(X, self.idf_, out=X)

        if self.norm:
            if self.norm == 'l1':
                norms = cp.abs(X).sum(axis=1)
            elif self.norm == 'l2':
                norms = cp.einsum('ij,ij->i', X, X)
                cp.sqrt(norms, out=norms)
            X = cp.divide(X, norms[:, cp.newaxis], out=X)

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
