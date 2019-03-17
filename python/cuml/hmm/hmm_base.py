from abc import ABC, abstractmethod
import numpy as np
from cuml.hmm.sample_utils import *

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

    @abstractmethod
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

    means_ = property(_get_means, _set_means)
    covars_ = property(_get_covars, _set_covars)
    weights_ = property(_get_weights, _set_weights)
    transmat_ = property(_get_transmat, _set_transmat)
