from cuml.hmm.core.hmm_base import _BaseHMM
from cuml.hmm.hmm_backend import _BaseHMMBackend

from cuml.gmm.sample_utils import *
from cuml.hmm.core.hmm_base import _BaseCUML


class _Multinomial(_BaseCUML):
    def __init__(self, init_params, precision, random_state):
        super().__init__(precision=precision, random_state=random_state)
        self.init_params = init_params

    def _initialize(self):
        if "e" in self.init_params :
            dPis = sample_matrix(1, self.n_features, random_state=self.random_state, isRowNorm=True)[0]
            self.set_emissionprob_(dPis)

    def _set_dims(self, X):
        self.n_features = np.max(X)

    def _setup(self, X):
        pass

    def get_emissionprob_(self):
        pis = self.dPis.copy_to_host()
        pis = pis.flatten()
        return pis

    def set_emissionprob_(self, prob):
        self.n_features = prob.shape[0]
        self.dPis = process_parameter(prob[:, None], self.n_features, self.dtype)

    _emissionprob_ = property(get_emissionprob_, set_emissionprob_)


class MultinomialHMM(_BaseHMM, _BaseHMMBackend):
    def __init__(self,
                 n_components,
                 init_params="ste",
                 precision="double",
                 random_state=None,
                 n_iter=10
                 ):

        _BaseHMM.__init__(self,
                          n_components=n_components,
                          precision=precision,
                          random_state=random_state,
                          init_params=init_params,
                          n_iter=n_iter)
        _BaseHMMBackend.__init__(self)

        self.x_type = np.uint16
        self.hmm_type = "multinomial"

        self.dists = [_Multinomial(init_params=self.init_params,
                                   precision=self.precision,
                                   random_state=self.random_state)
                      for _ in range(self.n_components)]

    def _get_emissionprob_(self):
        return np.array([dist._emissionprob_ for dist in self.dists])

    def _set_emissionprob_(self, emissionprob):
        for i in range(emissionprob.shape[0]) :
            self.dists[i]._emissionprob_ = emissionprob[i]

    emissionprob_ = property(_get_emissionprob_, _set_emissionprob_)
# TODO : Process and propagate int type