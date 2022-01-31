import cupy as cp
import numpy as np
import math
from cuml.common.input_utils import input_to_cupy_array
from cuml.common.base import Base
from cuml.metrics import pairwise_distances

VALID_KERNELS = [
    "gaussian",
    "tophat",
    "epanechnikov",
    "exponential",
    "linear",
    "cosine",
]


@cp.fuse()
def gaussian_log_kernel(x, h):
    return -(x*x)/(2*h*h)


@cp.fuse()
def tophat_log_kernel(x, h):
    ''' 
    if x < h:
        return 0.0
    else:
        return -FLOAT_MIN
    '''
    y = (x >= h)*x.dtype.type(1.0)
    y *= np.finfo(x.dtype).min
    return y


@cp.fuse()
def epanechnikov_log_kernel(x, h):
    # don't call log(0) otherwise we get NaNs
    z = cp.maximum(1.0 - (x * x) / (h * h), 1e-30)
    y = (x < h)*cp.log(z)
    y += (x >= h) * np.finfo(y.dtype).min
    return y


@cp.fuse()
def exponential_log_kernel(x, h):
    return -x/h


@cp.fuse()
def linear_log_kernel(x, h):
    # don't call log(0) otherwise we get NaNs
    z = cp.maximum(1.0 - x/h, 1e-30)
    y = (x < h)*cp.log(z)
    y += (x >= h) * np.finfo(y.dtype).min
    return y


@cp.fuse()
def cosine_log_kernel(x, h):
    # don't call log(0) otherwise we get NaNs
    z = cp.maximum(cp.cos(0.5*np.pi*x/h), 1e-30)
    y = (x < h)*cp.log(z)
    y += (x >= h) * np.finfo(y.dtype).min
    return y


def apply_log_kernel(distances, kernel, h):
    if kernel == "gaussian":
        return gaussian_log_kernel(distances, h)
    elif kernel == "tophat":
        return tophat_log_kernel(distances, h)
    elif kernel == "epanechnikov":
        return epanechnikov_log_kernel(distances, h)
    elif kernel == "exponential":
        return exponential_log_kernel(distances, h)
    elif kernel == "linear":
        return linear_log_kernel(distances, h)
    elif kernel == "cosine":
        return cosine_log_kernel(distances, h)
    else:
        raise ValueError("Unsupported kernel.")


def logVn(n):
    return 0.5*n*np.log(np.pi)-math.lgamma(0.5*n+1)


def logSn(n):
    return np.log(2*np.pi) + logVn(n - 1)


def norm_log_probabilities(log_probabilities, kernel, h, d):
    factor = 0.0
    if kernel == "gaussian":
        factor = 0.5*d*np.log(2*np.pi)
    elif kernel == "tophat":
        factor = logVn(d)
    elif kernel == "epanechnikov":
        factor = logVn(d) + np.log(2.0/(d+2.0))
    elif kernel == "exponential":
        factor = logSn(d - 1) + math.lgamma(d)
    elif kernel == "linear":
        factor = logVn(d) - np.log(d + 1.)
    elif kernel == "cosine":
        factor = 0.0
        tmp = 2. / np.pi
        for k in range(1, d + 1, 2):
            factor += tmp
            tmp *= -(d - k) * (d - k - 1) * (2. / np.pi) ** 2
        factor = np.log(factor) + logSn(d - 1)
    else:
        raise ValueError("Unsupported kernel.")

    return log_probabilities - (factor + d*np.log(h))


class KernelDensity(Base):
    """Kernel Density Estimation.
    Read more in the :ref:`User Guide <kernel_density>`.
    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth of the kernel.
    kernel : {'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', \
                 'cosine'}, default='gaussian'
        The kernel to use.
    metric : str, default='euclidean'
        The distance metric to use.  Note that not all metrics are
        valid with all algorithms. Note that the normalization of the density
        output is correct only for the Euclidean distance metric. Default
        is 'euclidean'.
    metric_params : dict, default=None
        Additional parameters to be passed to the tree for use with the
        metric.
    Examples
    --------
    Compute a gaussian kernel density estimate with a fixed bandwidth.
    >>> from sklearn.neighbors import KernelDensity
    >>> import numpy as np
    >>> rng = np.random.RandomState(42)
    >>> X = rng.random_sample((100, 3))
    >>> kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
    >>> log_density = kde.score_samples(X[:3])
    >>> log_density
    array([-1.52955942, -1.51462041, -1.60244657])
    """

    def __init__(
        self,
        *,
        bandwidth=1.0,
        kernel="gaussian",
        metric="euclidean",
        metric_params=None,
    ):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.metric = metric
        self.metric_params = metric_params or {}

        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        if kernel not in VALID_KERNELS:
            raise ValueError("invalid kernel: '{0}'".format(kernel))

    def fit(self, X, y=None, sample_weight=None):
        """Fit the Kernel Density model on the data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.
        sample_weight : array-like of shape (n_samples,), default=None
            List of sample weights attached to the data X.
            .. versionadded:: 0.20
        Returns
        -------
        self : object
            Returns the instance itself.
        """

        self.X_ = input_to_cupy_array(X, order='C', check_dtype=[cp.float32, cp.float64
                                                                 ]).array

        return self

    def score_samples(self, X):
        """Compute the log-likelihood of each sample under the model.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            An array of points to query.  Last dimension should match dimension
            of training data (n_features).
        Returns
        -------
        density : ndarray of shape (n_samples,)
            Log-likelihood of each sample in `X`. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """
        distances = pairwise_distances(
            X, self.X_, metric=self.metric, **self.metric_params)
        distances = cp.asarray(distances)
        h = self.bandwidth
        distances = apply_log_kernel(distances, self.kernel, h)
        log_probabilities = np.logaddexp.reduce(distances.get(), axis=1)

        # Note that sklearns user guide is wrong
        # It says the (unnormalised) probability output for the kernel density is sum(K(x,h))
        # In fact what they implment is (1/n)*sum(K(x,h))
        # Here we divide by n in normal probability space
        # Which becomes -log(n) in log probability space
        log_probabilities -= np.log(distances.shape[1])

        # norm
        log_probabilities = norm_log_probabilities(
            log_probabilities, self.kernel, h, X.shape[1])

        return log_probabilities

    def score(self, X, y=None):
        """Compute the total log-likelihood under the model.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.
        Returns
        -------
        logprob : float
            Total log-likelihood of the data in X. This is normalized to be a
            probability density, so the value will be low for high-dimensional
            data.
        """
        return cp.sum(self.score_samples(X))

    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from the model.
        Currently, this is implemented only for gaussian and tophat kernels.
        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.
        random_state : int, RandomState instance or None, default=None
            Determines random number generation used to generate
            random samples. Pass an int for reproducible results
            across multiple function calls.
            See :term:`Glossary <random_state>`.
        Returns
        -------
        X : array-like of shape (n_samples, n_features)
            List of samples.
        """
        pass
