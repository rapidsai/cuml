import numpy as np
from abc import ABC, abstractmethod


from cuml.gmm.sample_utils import sample_matrix
from hmmlearn import hmm
import hmmlearn
import cuml
from cuml.gmm.utils import *


class _Sampler(ABC):
    def __init__(self, n_seq, p=0.35, seq_len_ref=1, random_state=None):
        self.p = p
        self.random_state = random_state
        self.seq_len_ref = seq_len_ref

        self.n_seq = n_seq
        self.model = None

    def _sample_lengths(self):
        lengths = np.random.geometric(p=self.p, size=self.n_seq)
        lengths = [lengths[i] + self.seq_len_ref for i in range(len(lengths))]
        return lengths

    def sample_sequences(self):
        # Sample lengths
        lengths = self._sample_lengths()

        self.model = self._set_model_parameters(self.model, self.random_state)

        # Sample the sequences
        samples = [self.model.sample(lengths[sId])[0]
                   for sId in range(len(lengths))]
        samples = np.concatenate(samples, axis=0)

        if self.n_seq != 1 :
            lengths = np.array(lengths)
        else :
            lengths = None
        return samples, lengths

    def sample_startprob(self, random_state):
        return sample_matrix(1, self.n_components, random_state, isRowNorm=True)[0]

    def sample_transmat(self, random_state):
        return sample_matrix(self.n_components, self.n_components, random_state,
                                      isRowNorm=True)


class GMMHMMSampler(_Sampler):
    def __init__(self, n_seq, n_dim, n_mix, n_components,
                 p=0.35, seq_len_ref=5, random_state=None):
        super().__init__(n_seq, p, seq_len_ref, random_state)

        self.covariance_type = "full"

        self.n_dim = n_dim
        self.n_mix = n_mix
        self.n_components = n_components
        self.model = hmm.GMMHMM(n_components=self.n_components,
                                n_mix=self.n_mix,
                                covariance_type=self.covariance_type)

    def _sample_weights(self, random_state):
        weights = [sample_matrix(1, self.n_mix, random_state, isRowNorm=True)[0]
                      for _ in range(self.n_components)]
        return np.array(weights)

    def _sample_means(self, random_state):
        means = [[sample_matrix(1, self.n_dim, random_state)[0]
                      for _ in range(self.n_mix)]
                      for _ in range(self.n_components)]
        return np.array(means)[0]

    def _sample_covars(self, random_state):
            covars = [[sample_matrix(self.n_dim, self.n_dim, random_state, isSymPos=True)
                      for _ in range(self.n_mix)]
                      for _ in range(self.n_components)]
            return np.array(covars)

    def _set_model_parameters(self, model, random_state):
        model.n_features = self.n_dim

        model.startprob_ = self.sample_startprob(random_state)
        model.transmat_ = self.sample_transmat(random_state)
        model.means_ = self._sample_means(random_state)
        model.covars_ = self._sample_covars(random_state)
        model.weights_ = self._sample_weights(random_state)
        return model


class MultinomialHMMSampler(_Sampler):
    def __init__(self, n_seq, n_components, n_features,
                 p=0.35, seq_len_ref=5, random_state=None):
        super().__init__(n_seq, p, seq_len_ref, random_state)
        self.model = hmm.MultinomialHMM(n_components=n_components)
        self.n_components = n_components
        self.n_features = n_features

    def sample_emissionprob(self, random_state):
        return sample_matrix(self.n_components,
                             self.n_features,
                             random_state,
                             isRowNorm=True)

    def _set_model_parameters(self, model, random_state):
        model.startprob_ = self.sample_startprob(random_state)
        model.transmat_ = self.sample_transmat(random_state)
        model.emissionprob_ = self.sample_emissionprob(random_state)
        return model
