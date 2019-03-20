from cuml.hmm.base.hmm_base import _BaseHMM
from cuml.hmm.hmm_backend import _BaseHMMBackend
from cuml.gmm import GaussianMixture
from cuml.hmm._multinomial import _Multinomial

from cuml.gmm.sample_utils import *


class MultinomialHMM(_BaseHMM, _BaseHMMBackend):
    def __init__(self,
                 n_components,
                 precision,
                 random_state
                 ):

        _BaseHMM.__init__(self, precision=precision,
                          random_state=random_state)
        _BaseHMMBackend.__init__(self)

        self.n_components = n_components
        self.hmm_type = "multinomial"
        self.x_type = np.int32

    def _initialize(self):
        # Align flatten, cast and copy to device
        self.dT = sample_matrix(self.n_components, self.n_components, random_state=self.random_state, isRowNorm=True)
        self.lddt = roundup(self.n_components, self.align_size)
        self.dT = process_parameter(self.dT, self.lddt, self.dtype)

        self.dists = [_Multinomial(n_features=self.n_features,
                                   precision=self.precision,
                                   random_state=self.random_state)
                      for _ in range(self.n_components)]
        for dist in self.dists:
            dist._initialize()

    def _set_dims(self, X, lengths):
        self.nObs = X.shape[0]
        self.n_features = np.max(X)
        self.nStates = self.n_components

        if lengths is None:
            self.nSeq = 1
        else:
            self.nSeq = lengths.shape[0]

    def _setup(self, X, lengths):
        self.dB = sample_matrix(self.n_components,
                                self.nObs,
                                random_state=self.random_state,
                                isColNorm=True)
        self.lddb = roundup(self.n_components, self.align_size)
        self.dB = process_parameter(self.dB, self.lddb, self.dtype)

        self.dGamma = np.zeros((self.nStates, self.nObs), dtype=self.dtype)
        self.lddgamma = roundup(self.nStates, self.align_size)
        self.dGamma = process_parameter(self.dGamma, self.lddgamma, self.dtype)

        self.dX = X.T
        self.lddx = 1
        # Align flatten, cast and copy to device
        self.dX = process_parameter(self.dX, self.lddx, self.x_type)

        # Process lengths
        if lengths is None:
            lengths = np.array([self.nObs])
        # Check leading dimension
        lengths = lengths.astype(int)
        self.dlengths = cuda.to_device(lengths)

    def _get_emissionprob_(self):
        return np.array([dist.emissionprob_ for dist in self.dists])

    def _set_emissionprob_(self):
        pass

    emissionprob_ = property(_get_emissionprob_, _set_emissionprob_)

# TODO : Process and propagate int type