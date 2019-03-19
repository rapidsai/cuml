from abc import ABC, abstractmethod
from cuml.gmm.sample_utils import *

from cuml.gmm import GaussianMixture

RUP_SIZE = 32

class _BaseCUML(ABC):
    def _get_ctype_ptr(self, obj):
        return obj.device_ctypes_pointer.value

    def _get_column_ptr(self, obj):
        return self._get_ctype_ptr(obj._column._data.to_gpu_array())

    def _get_dtype(self, precision):
        return {
            'single': np.float32,
            'double': np.float64,
        }[precision]

    def __init__(self, precision):
        self.precision = precision
        self.dtype = self._get_dtype(precision)


class _BaseGMM(_BaseCUML):
    def __init__(self, precision):
        super().__init__(precision=precision)

    def initialize(self):
        pass

class _BaseHMM(_BaseCUML):
    def __init__(self,
                 n_components,
                 n_mix,
                 precision,
                 random_state
                 ):

        super().__init__(precision=precision)

        self.n_components = n_components
        self.n_mix = n_mix
        self.random_state=random_state

    def fit(self, X, lengths=None):
        self._fit(X, lengths)

    # @abstractmethod
    # def decode(self, X, lengths=None, algorithm=None):
    #     pass

    # @abstractmethod
    # def predict(self, X, lengths=None):
    #     pass

    def predict_proba(self, X, lengths=None):
        self._forward_backard(X, lengths, True, True)
        self._compute_gammas(X, lengths)
        return self._gammas_

    # @abstractmethod
    # def sample(self, n_samples=1, random_state=None):
    #     pass

    def score(self, X, lengths=None):
        self._forward_backard(X, lengths, True, False)
        return self._score()

    def score_samples(self, X, lengths=None):
        logprob = self.score(X, lengths)
        posteriors = self.predict_proba(X, lengths)
        return logprob, posteriors

    # TODO : Fix setters
    def _get_means(self):
        return np.array([gmm.means_ for gmm in self.gmms])

    def _set_means(self, means):
        for i in range(len(means.shape[0])):
            self.gmms[i].set_means(means[i])

    def _get_covars(self):
        return np.array([gmm.covariances_ for gmm in self.gmms])

    def _set_covars(self, covars):
        for i in range(len(covars.shape[0])):
            self.gmms[i].set_means(covars[i])

    def _get_weights(self):
        return np.array([gmm.weights_ for gmm in self.gmms])

    def _set_weights(self, weights):
        for i in range(len(weights.shape[0])):
            self.gmms[i].set_means(weights[i])

    def _get_transmat(self):
        T = self.dT.copy_to_host()
        T = deallign(T, self.nStates, self.nStates, self.lddt)
        return T

    def _set_transmat(self, transmat):
        for i in range(len(transmat.shape[0])):
            self.gmms[i].set_means(transmat[i])

    def _get_gamma(self):
        gamma = self.dGamma.copy_to_host()
        gamma = deallign(gamma, self.nStates, self.nObs, self.lddgamma)
        return gamma

    def _set_gamma(self, gamma):
        pass

    means_ = property(_get_means, _set_means)
    covars_ = property(_get_covars, _set_covars)
    weights_ = property(_get_weights, _set_weights)
    transmat_ = property(_get_transmat, _set_transmat)
    _gamma_ = property(_get_gamma, _set_gamma)

class _GMMHMM(_BaseHMM):
    def __init__(self,
                 n_components,
                 n_mix,
                 precision,
                 random_state
                 ):

        super().__init__(n_components=n_components,
                 n_mix=n_mix,
                 precision=precision,
                 random_state=random_state)

    def _initialize(self):
        # Align flatten, cast and copy to device
        self.dT = sample_matrix(self.n_components, self.n_mix, random_state=self.random_state, isRowNorm=True)
        self.lddt = roundup(self.n_components, RUP_SIZE)
        self.dT = process_parameter(self.dT, self.lddt, self.dtype)

        self.gmms = [GaussianMixture(n_components=self.nCl,
                                     precision=self.precision) for _ in range(self.n_components)]
        for gmm in self.gmms:
            gmm._set_dims(nCl=self.nCl, nDim=self.nDim, nObs=self.nObs)
            gmm._initialize()

    def _set_dims(self, X, lengths):
        self.nObs = X.shape[0]
        self.nDim = X.shape[1]
        self.nCl = self.n_mix
        self.nStates = self.n_components

        if lengths is None :
            self.n_seq = 1
        else :
            self.n_seq = lengths.shape[0]

    def _setup(self, X, lengths):
        self.dB = sample_matrix(self.n_components,
                                self.nObs * self.nCl,
                                random_state=self.random_state,
                                isColNorm=True)
        self.lddb = roundup(self.nCl, RUP_SIZE)
        self.dB = process_parameter(self.dB, self.lddb, self.dtype)

        self.dGamma = np.zeros((self.nStates, self.nObs), dtype=self.dtype)
        self.lddgamma = roundup(self.nStates, RUP_SIZE)
        self.dGamma = process_parameter(self.dGamma, self.lddgamma, self.dtype)

        for gmm in self.gmms :
            gmm._setup(X)

        self.dX = X.T
        self.lddx = roundup(self.nDim, RUP_SIZE)
        # Align flatten, cast and copy to device
        self.dX = process_parameter(self.dX, self.lddx, self.dtype)

        # Process lengths
        if lengths is None :
            lengths = np.array([self.nObs])
        # Check leading dimension
        lengths = lengths.astype(int)
        self.dlengths = cuda.to_device(lengths)