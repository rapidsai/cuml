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
        self.dParams = None

    def initialize(self):
        pass



class _BaseHMM(_BaseCUML):
    def __init__(self,
                 n_components=1,
                 n_mix=1,
                 min_covar=0.001,
                 startprob_prior=1.0,
                 transmat_prior=1.0,
                 weights_prior=1.0,
                 means_prior=0.0,
                 means_weight=0.0,
                 covars_prior=None,
                 covars_weight=None,
                 algorithm='viterbi',
                 covariance_type='diag',
                 random_state=None,
                 n_iter=10,
                 tol=0.01,
                 verbose=False,
                 params='stmcw',
                 init_params='stmcw',
                 precision=np.float64
                 ):

        super().__init__(precision=precision)

        self.n_components = n_components
        self.n_mix = n_mix
        self.min_covar = min_covar
        self.startprob_prior = startprob_prior
        self.transmat_prior = transmat_prior
        self.weights_prior = weights_prior
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight
        self.algorithm = algorithm
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.params = params
        self.init_params = init_params

        self.gmms = [_BaseGMM(self.precision) for _ in range(self.n_mix)]

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def decode(self, X, lengths=None, algorithm=None):
        pass

    @abstractmethod
    def predict(self, X, lengths=None):
        pass

    @abstractmethod
    def predict_proba(self, X, lengths=None):
        pass

    @abstractmethod
    def sample(self, n_samples=1, random_state=None)
        pass

    @abstractmethod
    def score(self, X, lengths=None):
        pass

    @abstractmethod
    def score_samples(self, X, lengths=None):
        pass

    def _setup(self, X, lengths=None):
        params = dict()
        params["T"] = sample_matrix(self.n_components, self.n_mix, isRowNorm=True)

    def _convert(self):
        pass

    @property
    def means_(self):
        return np.array([gmm.means_ for gmm in self.gmms])

    @property
    def covars_(self):
        return np.array([gmm.covariances_ for gmm in self.gmms])

    @property
    def weights_(self):
        return np.array([gmm.weights_ for gmm in self.gmms])
