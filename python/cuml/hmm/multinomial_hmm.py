from cuml.hmm.base.hmm_base import _BaseHMM
from cuml.hmm.hmm_backend import _BaseHMMBackend
from cuml.gmm import GaussianMixture
from cuml.hmm.multinomial.multinomial import Multinomial

from cuml.gmm.sample_utils import *


class _GMMHMM(_BaseHMM, _BaseHMMBackend):
    def __init__(self,
                 n_components,
                 precision,
                 random_state
                 ):

        _BaseHMM.__init__(precision=precision,
                          random_state=random_state)
        self.n_components = n_components

        self.hmm_type = "multinomial"

    def _initialize(self):
        # Align flatten, cast and copy to device
        self.dT = sample_matrix(self.n_components, self.n_mix, random_state=self.random_state, isRowNorm=True)
        self.lddt = roundup(self.n_components, self.align_size)
        self.dT = process_parameter(self.dT, self.lddt, self.dtype)

        self.dists = [GaussianMixture(n_components=self.nCl,
                                      precision=self.precision) for _ in range(self.n_components)]
        for dist in self.dists:
            dist._set_dims(nCl=self.nCl, nDim=self.nDim, nObs=self.nObs)
            dist._initialize()

    def _set_dims(self, X, lengths):
        self.nObs = X.shape[0]
        self.nDim = X.shape[1]
        self.nCl = self.n_mix
        self.nStates = self.n_components

        if lengths is None:
            self.nSeq = 1
        else:
            self.nSeq = lengths.shape[0]

    def _setup(self, X, lengths):
        self.dB = sample_matrix(self.n_components,
                                self.nObs * self.nCl,
                                random_state=self.random_state,
                                isColNorm=True)
        self.lddb = roundup(self.nCl, self.align_size)
        self.dB = process_parameter(self.dB, self.lddb, self.dtype)

        self.dGamma = np.zeros((self.nStates, self.nObs), dtype=self.dtype)
        self.lddgamma = roundup(self.nStates, self.align_size)
        self.dGamma = process_parameter(self.dGamma, self.lddgamma, self.dtype)

        for dist in self.dists:
            dist._setup(X)

        self.dX = X.T
        self.lddx = roundup(self.nDim, self.align_size)
        # Align flatten, cast and copy to device
        self.dX = process_parameter(self.dX, self.lddx, self.dtype)

        # Process lengths
        if lengths is None:
            lengths = np.array([self.nObs])
        # Check leading dimension
        lengths = lengths.astype(int)
        self.dlengths = cuda.to_device(lengths)