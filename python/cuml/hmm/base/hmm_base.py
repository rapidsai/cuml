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

    def __init__(self, precision, random_state, align_size=32):
        self.precision = precision
        self.dtype = self._get_dtype(precision)
        self.random_state = random_state

        self.align_size = align_size

class _BaseGMM(_BaseCUML):
    def __init__(self, precision):
        super().__init__(precision=precision)

    def initialize(self):
        pass

class _BaseHMM(_BaseCUML, _DevHMM):
    def __init__(self,
                 precision,
                 random_state,
                 init_params
                 ):

        _BaseCUML.__init__(self,
                           precision=precision,
                           random_state=random_state)
        self.init_params = init_params

    def fit(self, X, lengths=None):
        self._fit(X, lengths)

    def decode(self, X, lengths=None, algorithm=None):
        self._viterbi(X, lengths)
        state_sequence = self._dVStates_
        llhd = self._llhd
        return llhd, state_sequence

    # @abstractmethod
    # def predict(self, X, lengths=None):
    #     pass

    def predict_proba(self, X, lengths=None):
        self._forward_backward(X, lengths, True, True, True)
        return self._gammas_

    # @abstractmethod
    # def sample(self, n_samples=1, random_state=None):
    #     pass

    def score(self, X, lengths=None):
        self._forward_backward(X, lengths, True, False, False)
        return self._score()

    def score_samples(self, X, lengths=None):
        posteriors = self.predict_proba(X, lengths)
        logprob = self._score()
        return logprob, posteriors


    def _get_transmat(self):
        T = self.dT.copy_to_host()
        T = deallign(T, self.nStates, self.nStates, self.lddt)
        return T

    def _set_transmat(self, transmat):
        self.lddt = roundup(self.n_components, self.align_size)
        self.dT = process_parameter(transmat, self.lddt, self.dtype)

    def _get_emissionprob_(self):
        return np.array([dist._emissionprob_ for dist in self.dists])

    def _set_emissionprob_(self, emissionprob):
        for i in range(emissionprob.shape[0]) :
            self.dists[i]._emissionprob_ = emissionprob[i]

    def _get_startprob_(self):
        startProb = self.dstartProb.copy_to_host()
        return startProb

    def _set_startprob_(self, startProb):
        self.lddsp = startProb.shape[0]
        self.dstartProb = process_parameter(startProb[:, None], self.lddsp, self.dtype)

    transmat_ = property(_get_transmat, _set_transmat)
    emissionprob_ = property(_get_emissionprob_, _set_emissionprob_)
    startprob_ = property(_get_startprob_, _set_startprob_)
