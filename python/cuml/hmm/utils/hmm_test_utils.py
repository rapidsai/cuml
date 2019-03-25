import numpy as np
from abc import ABC, abstractmethod


from cuml.gmm.sample_utils import sample_matrix
from hmmlearn import hmm
import hmmlearn
import cuml
from cuml.gmm.utils import *

class _Sampler(ABC):
    def __init__(self, n_seq, p=0.35, seq_len_ref=5, random_state=None):
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

        self._set_model_parameters()

        # Sample the sequences
        samples = [self.model.sample(lengths[sId])[0]
                   for sId in range(len(lengths))]
        samples = np.concatenate(samples, axis=0)
        lengths = np.array(lengths)
        return samples, lengths

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

    def _sample_weights(self):
        weights = [sample_matrix(1, self.n_mix, self.random_state, isRowNorm=True)[0]
                      for _ in range(self.n_components)]
        return np.array(weights)

    def _sample_means(self):
        means = [[sample_matrix(1, self.n_dim, self.random_state)[0]
                      for _ in range(self.n_mix)]
                      for _ in range(self.n_components)]
        return np.array(means)

    def _sample_covars(self):
            covars = [[sample_matrix(self.n_dim, self.n_dim, self.random_state, isSymPos=True)
                      for _ in range(self.n_mix)]
                      for _ in range(self.n_components)]
            return np.array(covars)

    def _set_model_parameters(self):
        self.model.n_features = self.n_dim

        self.model.startprob_ = sample_matrix(1, self.n_components, self.random_state, isRowNorm=True)[0]
        self.model.transmat_ = sample_matrix(self.n_components, self.n_components, self.random_state,
                                      isRowNorm=True)
        self.model.means_ = self._sample_means()
        self.model.covars_ = self._sample_covars()
        self.model.weights_ = self._sample_weights()


class MultinomialHMMSampler(_Sampler):
    def __init__(self, n_seq, n_components, n_features,
                 p=0.35, seq_len_ref=5, random_state=None):
        super().__init__(n_seq, p, seq_len_ref, random_state)
        self.model = hmm.MultinomialHMM(n_components=n_components)
        self.n_components = n_components
        self.n_features = n_features

    def _set_model_parameters(self):
        self.model.startprob_ = sample_matrix(1, self.n_components, self.random_state, isRowNorm=True)[0]
        self.model.transmat_ = sample_matrix(self.n_components, self.n_components, self.random_state,
                                      isRowNorm=True)
        self.model.emissionprob_ = sample_matrix(self.n_components, self.n_features, self.random_state,
                                      isRowNorm=True)


class _Tester(ABC):
    def __init__(self, precision, random_state):
        self.precision = precision
        self.random_state = random_state

    def test_workflow(self, X, lengths):
        self._reset()


        self.cuml_model._forward_backward(X, lengths, True, False)
        print(self.cuml_model.emissionprob_)

    def test_score_samples(self, X, lengths):
        sk_out = self.ref_model.score_samples(X, lengths)
        cuml_out = self.cuml_model.score_samples(X, lengths)
        return mae(cuml_out, sk_out)

# class GMMHMMTester(_Tester) :
#     def __init__(self, n_components):
#         super().__init__()
#         self.n_components =n_components
#
#     def _reset(self):
#             self.sk = hmmlearn.hmm.GMMHMM(self.n_components)
#             self.cuml = cuml.hmm.GMMHMM(self.n_components)

class MultinomialHMMTester(_Tester):
    def __init__(self, n_components, precision, random_state):
        super().__init__(precision, random_state)
        self.n_components = n_components

    def _reset(self):
        self.ref_model = hmmlearn.hmm.MultinomialHMM(self.n_components)
        self.cuml_model = cuml.hmm.MultinomialHMM(self.n_components,
                                            precision=self.precision,
                                            random_state=self.random_state)

