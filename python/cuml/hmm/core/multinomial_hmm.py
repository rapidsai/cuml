from cuml.hmm.core.hmm_base import _BaseHMM
from cuml.hmm.hmm_backend import _BaseHMMBackend
from cuml.hmm._multinomial import _Multinomial

from cuml.gmm.sample_utils import *


class MultinomialHMM(_BaseHMM, _BaseHMMBackend):
    def __init__(self,
                 n_components,
                 init_params="ste",
                 precision="double",
                 random_state=None,
                 n_iter=10
                 ):

        _BaseHMM.__init__(self, precision=precision,
                          random_state=random_state,
                          init_params=init_params,
                          n_iter=n_iter)
        _BaseHMMBackend.__init__(self)

        self.n_components = n_components
        self.x_type = np.int32

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