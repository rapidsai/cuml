import numpy as np

from cuml.hmm.sample_utils import sample_matrix
from hmmlearn import hmm

startprob = np.array([0.6, 0.3, 0.1, 0.0])
# The transition matrix, note that there are no transitions possible
# between component 1 and 3
transmat = np.array([[0.7, 0.2, 0.0, 0.1],
                     [0.3, 0.5, 0.2, 0.0],
                     [0.0, 0.3, 0.5, 0.2],
                     [0.2, 0.0, 0.2, 0.6]])
# The means of each component
means = np.array([[0.0,  0.0],
                  [0.0, 11.0],
                  [9.0, 10.0],
                  [11.0, -1.0]])
# The covariance of each component
covars = .5 * np.tile(np.identity(2), (4, 1, 1))


class HMMSampler :
    def __init__(self, n_seq, n_dim, n_mix, n_components,
                 p=0.35, seq_len_ref=5, random_state=None):
        self.p = p
        self.random_state=random_state
        self.seq_len_ref = seq_len_ref
        self.covariance_type = "full"

        self.n_seq = n_seq
        self.n_dim = n_dim
        self.n_mix = n_mix
        self.n_components = n_components

    def _sample_lengths(self):
        lengths = np.random.geometric(p=self.p, size=self.n_seq)
        lengths = [lengths[i] + self.seq_len_ref for i in range(len(lengths))]
        return lengths

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

    def sample_sequences(self):
        # Sample lengths
        lengths = self._sample_lengths()

        # Set HMM model
        self.model = hmm.GMMHMM(n_components=self.n_components,
                                     n_mix=self.n_mix,
                                covariance_type=self.covariance_type)
        self._set_model_parameters()

        # Sample the sequences
        samples = [self.model.sample(lengths[sId])[0]
                   for sId in range(len(lengths))]
        samples = np.concatenate(samples, axis=0)
        lengths = np.array(lengths)
        return samples, lengths

