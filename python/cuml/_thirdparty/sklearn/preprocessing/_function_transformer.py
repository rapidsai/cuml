# This code originates from the Scikit-Learn library,
# it was since modified to allow GPU acceleration.
# This code is under BSD 3 clause license.
# Authors mentioned above do not endorse or promote this production.


import warnings

import cuml
from ....common.array_sparse import SparseCumlArray
from ..utils.skl_dependencies import TransformerMixin, BaseEstimator
from ..utils.validation import _allclose_dense_sparse
from ....internals import _deprecate_pos_args


def _identity(X):
    """The identity function.
    """
    return X


class FunctionTransformer(TransformerMixin, BaseEstimator):
    """Constructs a transformer from an arbitrary callable.

    A FunctionTransformer forwards its X (and optionally y) arguments to a
    user-defined function or function object and returns the result of this
    function. This is useful for stateless transformations such as taking the
    log of frequencies, doing custom scaling, etc.

    Note: If a lambda is used as the function, then the resulting
    transformer will not be pickleable.

    Parameters
    ----------
    func : callable, default=None
        The callable to use for the transformation. This will be passed
        the same arguments as transform, with args and kwargs forwarded.
        If func is None, then func will be the identity function.

    inverse_func : callable, default=None
        The callable to use for the inverse transformation. This will be
        passed the same arguments as inverse transform, with args and
        kwargs forwarded. If inverse_func is None, then inverse_func
        will be the identity function.

    accept_sparse : bool, default=False
        Indicate that func accepts a sparse matrix as input. Otherwise,
        if accept_sparse is false, sparse matrix inputs will cause
        an exception to be raised.

    check_inverse : bool, default=True
       Whether to check that or ``func`` followed by ``inverse_func`` leads to
       the original inputs. It can be used for a sanity check, raising a
       warning when the condition is not fulfilled.

    kw_args : dict, default=None
        Dictionary of additional keyword arguments to pass to func.

    inv_kw_args : dict, default=None
        Dictionary of additional keyword arguments to pass to inverse_func.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.preprocessing import FunctionTransformer
    >>> transformer = FunctionTransformer(cp.log1p)
    >>> X = cp.array([[0, 1], [2, 3]])
    >>> transformer.transform(X)
    array([[0.       , 0.6931...],
           [1.0986..., 1.3862...]])
    """

    @_deprecate_pos_args(version="0.20")
    def __init__(self, *, func=None, inverse_func=None, accept_sparse=False,
                 check_inverse=True, kw_args=None, inv_kw_args=None):
        self.func = func
        self.inverse_func = inverse_func
        self.accept_sparse = accept_sparse
        self.check_inverse = check_inverse
        self.kw_args = kw_args
        self.inv_kw_args = inv_kw_args

    def _check_input(self, X):
        return self._validate_data(X, accept_sparse=self.accept_sparse)

    def _check_inverse_transform(self, X):
        """Check that func and inverse_func are the inverse."""
        interval = max(1, X.shape[0] // 100)
        selection = [i * interval for i in range(X.shape[0] // interval)]
        with cuml.using_output_type("cupy"):
            X_round_trip = self.inverse_transform(self.transform(X[selection]))
            if not _allclose_dense_sparse(X[selection], X_round_trip):
                warnings.warn("The provided functions are not strictly"
                              " inverse of each other. If you are sure you"
                              " want to proceed regardless, set"
                              " 'check_inverse=False'.", UserWarning)

    def fit(self, X, y=None) -> "FunctionTransformer":
        """Fit transformer by checking X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input array.

        Returns
        -------
        self
        """
        X = self._check_input(X)
        if (self.check_inverse and not (self.func is None or
                                        self.inverse_func is None)):
            self._check_inverse_transform(X)
        return self

    def transform(self, X) -> SparseCumlArray:
        """Transform X using the forward function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input array.

        Returns
        -------
        X_out : {array-like, sparse matrix}, shape (n_samples, n_features)
            Transformed input.
        """
        return self._transform(X, func=self.func, kw_args=self.kw_args)

    def inverse_transform(self, X) -> SparseCumlArray:
        """Transform X using the inverse function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input array.

        Returns
        -------
        X_out : {array-like, sparse matrix}, shape (n_samples, n_features)
            Transformed input.
        """
        return self._transform(X, func=self.inverse_func,
                               kw_args=self.inv_kw_args)

    def _transform(self, X, func=None, kw_args=None):
        X = self._check_input(X)

        if func is None:
            func = _identity

        return func(X, **(kw_args if kw_args else {}))

    def _more_tags(self):
        return {'stateless': True,
                'requires_y': False}
