import numpy as np
import matplotlib.pyplot as plt

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



def sample_sequences(startprob, transmat,  means, covars, lengths,
                     n_components, covariance_type="full"):

    model = hmm.GaussianHMM(n_components=n_components,
                            covariance_type=covariance_type)

    model.startprob_ = startprob
    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = covars

    # Generate samples
    samples = [model.sample(lengths[sId])[0] for sId in range(len(lengths))]
    return samples