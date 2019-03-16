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

    # @abstractmethod
    # def fit(self, X, lengths=None):
    #     pass
    #
    # @abstractmethod
    # def decode(self, X, lengths=None, algorithm=None):
    #     pass
    #
    # @abstractmethod
    # def predict(self, X, lengths=None):
    #     pass
    #
    # @abstractmethod
    # def predict_proba(self, X, lengths=None):
    #     pass
    #
    # @abstractmethod
    # def sample(self, n_samples=1, random_state=None):
    #     pass
    #
    # @abstractmethod
    # def score(self, X, lengths=None):
    #     pass
    #
    # @abstractmethod
    # def score_samples(self, X, lengths=None):
    #     pass

    @property
    def means_(self):
        return np.array([gmm.means_ for gmm in self.gmms])

    @property
    def covars_(self):
        return np.array([gmm.covariances_ for gmm in self.gmms])

    @property
    def weights_(self):
        return np.array([gmm.weights_ for gmm in self.gmms])

