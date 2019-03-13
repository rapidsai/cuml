import warnings
from abc import ABCMeta, abstractmethod
from time import time

import numpy as np

from .. import cluster
from sklearn.base import BaseEstimator
from sklearn.base import DensityMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array, check_random_state
from sklearn.utils.fixes import logsumexp
from sklearn.mixture.base import _check_X
from sklearn.mixture.gaussian_mixture import _estimate_gaussian_parameters, _compute_precision_cholesky

from scipy import linalg


def _initialize(self, X, resp):
    """Initialization of the Gaussian mixture parameters.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    resp : array-like, shape (n_samples, n_components)
    """
    n_samples, _ = X.shape

    weights, means, covariances = _estimate_gaussian_parameters(
        X, resp, self.reg_covar, self.covariance_type)
    weights /= n_samples

    print("resp at initialization", resp)
    print("\n")

    self.weights_ = (weights if self.weights_init is None
                     else self.weights_init)
    self.means_ = means if self.means_init is None else self.means_init

    if self.precisions_init is None:
        self.covariances_ = covariances
        self.precisions_cholesky_ = _compute_precision_cholesky(
            covariances, self.covariance_type)
    elif self.covariance_type == 'full':
        self.precisions_cholesky_ = np.array(
            [linalg.cholesky(prec_init, lower=True)
             for prec_init in self.precisions_init])
    elif self.covariance_type == 'tied':
        self.precisions_cholesky_ = linalg.cholesky(self.precisions_init,
                                                    lower=True)
    else:
        self.precisions_cholesky_ = self.precisions_init


def _initialize_parameters(self, X, random_state):
    """Initialize the model parameters.
    Parameters
    ----------
    X : array-like, shape  (n_samples, n_features)
    random_state : RandomState
        A random number generator instance.
    """
    n_samples, _ = X.shape

    if self.init_params == 'kmeans':
        resp = np.zeros((n_samples, self.n_components))
        label = cluster.KMeans(n_clusters=self.n_components, n_init=1,
                               random_state=random_state).fit(X).labels_
        resp[np.arange(n_samples), label] = 1
    elif self.init_params == 'random':
        resp = random_state.rand(n_samples, self.n_components)
        resp /= resp.sum(axis=1)[:, np.newaxis]
    else:
        raise ValueError("Unimplemented initialization method '%s'"
                         % self.init_params)

    _initialize(self, X, resp)

def fit_predict(self, X, y=None):
    """Estimate model parameters using X and predict the labels for X.
    The method fits the model n_init times and sets the parameters with
    which the model has the largest likelihood or lower bound. Within each
    trial, the method iterates between E-step and M-step for `max_iter`
    times until the change of likelihood or lower bound is less than
    `tol`, otherwise, a `ConvergenceWarning` is raised. After fitting, it
    predicts the most probable label for the input data points.
    .. versionadded:: 0.20
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row
        corresponds to a single data point.
    Returns
    -------
    labels : array, shape (n_samples,)
        Component labels.
    """
    X = _check_X(X, self.n_components, ensure_min_samples=2)
    self._check_initial_parameters(X)

    # if we enable warm_start, we will have a unique initialisation
    do_init = not (self.warm_start and hasattr(self, 'converged_'))
    n_init = self.n_init if do_init else 1

    max_lower_bound = -np.infty
    self.converged_ = False

    random_state = check_random_state(self.random_state)

    n_samples, _ = X.shape
    for init in range(n_init):
        self._print_verbose_msg_init_beg(init)

        if do_init:
            _initialize_parameters(self, X, random_state)

        lower_bound = (-np.infty if do_init else self.lower_bound_)


        for n_iter in range(1, self.max_iter + 1):
            prev_lower_bound = lower_bound

            log_prob_norm, log_resp = self._e_step(X)

            self._m_step(X, log_resp)
            lower_bound = self._compute_lower_bound(
                log_resp, log_prob_norm)

            change = lower_bound - prev_lower_bound
            self._print_verbose_msg_iter_end(n_iter, change)

            # if abs(change) < self.tol:
            #     self.converged_ = True
            #     break
            #
            print('iter', n_iter, "\n")
            print(np.exp(log_resp))
            print('\n')

            print(lower_bound)


        self._print_verbose_msg_init_end(lower_bound)

        if lower_bound > max_lower_bound:
            max_lower_bound = lower_bound
            best_params = self._get_parameters()
            best_n_iter = n_iter

    if not self.converged_:
        warnings.warn('Initialization %d did not converge. '
                      'Try different init parameters, '
                      'or increase max_iter, tol '
                      'or check for degenerate data.'
                      % (init + 1), ConvergenceWarning)

    self._set_parameters(best_params)
    self.n_iter_ = best_n_iter
    self.lower_bound_ = max_lower_bound

    # Always do a final e-step to guarantee that the labels returned by
    # fit_predict(X) are always consistent with fit(X).predict(X)
    # for any value of max_iter and tol (and any random_state).
    _, log_resp = self._e_step(X)

    return log_resp.argmax(axis=1)