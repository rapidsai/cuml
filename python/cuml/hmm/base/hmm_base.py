from abc import ABC, abstractmethod
from cuml.gmm.sample_utils import *


from cuml.hmm.utils.devtools import _DevHMM

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

    def __init__(self, precision, align_size=32):
        self.precision = precision
        self.dtype = self._get_dtype(precision)

        self.align_size = align_size

class _BaseGMM(_BaseCUML):
    def __init__(self, precision):
        super().__init__(precision=precision)

    def initialize(self):
        pass

class _BaseHMM(_BaseCUML, _DevHMM):
    def __init__(self,
                 precision,
                 random_state
                 ):

        _BaseCUML.__init__(precision=precision)
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


    def _get_transmat(self):
        T = self.dT.copy_to_host()
        T = deallign(T, self.nStates, self.nStates, self.lddt)
        return T

    def _set_transmat(self, transmat):
        for i in range(len(transmat.shape[0])):
            self.dists[i].set_means(transmat[i])

    def _get_gamma(self):
        gamma = self.dGamma.copy_to_host()
        gamma = deallign(gamma, self.nStates, self.nObs, self.lddgamma)
        return gamma

    def _set_gamma(self, gamma):
        pass


    transmat_ = property(_get_transmat, _set_transmat)
    _gamma_ = property(_get_gamma, _set_gamma)
    

