from cuml.hmm.core.hmm_base import _BaseHMM
from cuml.hmm.hmm_backend import _BaseHMMBackend
from cuml.gmm import GaussianMixture

import numpy as np


class GMMHMM(_BaseHMM, _BaseHMMBackend):
    def __init__(self,
                 n_components,
                 n_mix,
                 precision='double',
                 random_state=None,
                 init_params="ste",
                 n_iter=10
                 ):

        _BaseHMM.__init__(self,
                          n_components=n_components,
                          precision=precision,
                          random_state=random_state,
                          init_params=init_params,
                          n_iter=n_iter)
        self.n_components = n_components
        self.n_mix = n_mix
        self.hmm_type = "gmm"

        self.x_type = np.int32

        self.dists = [GaussianMixture(init_params=self.init_params,
                                   precision=self.precision,
                                   random_state=self.random_state)
                      for _ in range(self.n_components)]

    # TODO : Fix setters
    def _get_means(self):
        return np.array([dist.means_ for dist in self.dists])

    def _set_means(self, means):
        for i in range(len(means.shape[0])):
            self.dists[i].set_means(means[i])

    def _get_covars(self):
        return np.array([dist.covariances_ for dist in self.dists])

    def _set_covars(self, covars):
        for i in range(len(covars.shape[0])):
            self.dists[i].set_means(covars[i])

    def _get_weights(self):
        return np.array([dist.weights_ for dist in self.dists])

    def _set_weights(self, weights):
        for i in range(len(weights.shape[0])):
            self.dists[i].set_means(weights[i])

    means_ = property(_get_means, _set_means)
    covars_ = property(_get_covars, _set_covars)
    weights_ = property(_get_weights, _set_weights)