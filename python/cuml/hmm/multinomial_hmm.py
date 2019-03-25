from cuml.hmm.base.hmm_base import _BaseHMM
from cuml.hmm.hmm_backend import _BaseHMMBackend
from cuml.gmm import GaussianMixture
from cuml.hmm._multinomial import _Multinomial

from cuml.gmm.sample_utils import *


class MultinomialHMM(_BaseHMM, _BaseHMMBackend):
    def __init__(self,
                 n_components,
                 init_params="ste",
                 precision="double",
                 random_state=None
                 ):

        _BaseHMM.__init__(self, precision=precision,
                          random_state=random_state,
                          init_params=init_params)
        _BaseHMMBackend.__init__(self)

        self.n_components = n_components
        self.hmm_type = "multinomial"
        self.x_type = np.int32

        self.dists = [_Multinomial(init_params=self.init_params,
            precision=self.precision,
                                   random_state=self.random_state)
                      for _ in range(self.n_components)]

    def _initialize(self):
        # Align flatten, cast and copy to device
        if 't' in self.init_params :
            init_value = 1 / self.n_components
            T = np.full((self.n_components, self.n_components), init_value)
            self._set_transmat(T)

        if 's' in self.init_params :
            init_value = 1 / self.n_components
            sp = np.full(self.n_components, init_value)
            self._set_startprob_(sp)

        for dist in self.dists:
            dist._initialize()

    def _set_dims(self, X, lengths):
        self.nObs = X.shape[0]
        self.nStates = self.n_components
        # self.nFeatures = np.max(X)

        if lengths is None:
            self.nSeq = 1
        else:
            self.nSeq = lengths.shape[0]

        for dist in self.dists :
            dist._set_dims(X)

    def _setup(self, X, lengths):

        B = sample_matrix(self.n_components,
                                self.nObs,
                                random_state=self.random_state,
                                isColNorm=True)
        self._set_B(B)

        Gamma = np.zeros((self.nStates, self.nObs), dtype=self.dtype)
        self._set_gamma(Gamma)

        Llhd = np.zeros(self.nSeq, dtype=self.dtype)
        self._set_llhd(Llhd)

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

        for dist in self.dists :
            dist._setup(X)

    def _score(self):
        return sum(self._llhd)


# TODO : Process and propagate int type