from cuml.hmm.utils.utils import GMMHMMSampler, MultinomialHMMSampler
import cuml
import os
import numpy as np

def to_csv(data_dict, path):
    for key, data in data_dict :
        full_path = os.path.join(path, key)
        np.savetxt(data, full_path)


if __name__ == '__main__' :
    n_seq = 10
    n_dim=3
    n_mix=3,
    n_components=5
    precision = 'double'
    random_state = None
    path = "/home/mehdia/repos/build-rapids/cuml/cuML/profile/data"

    hmm_sampler = GMMHMMSampler(n_seq, n_dim, n_mix, n_components)
    cuml_model = cuml.hmm.GMMHMM(n_components,
                                         precision=precision,
                                         random_state=random_state)

    X, lengths = hmm_sampler.sample_sequences()
    cuml_model.score_samples(X, lengths)
    data_dict = {"means" : cuml_model.means_,
                 "covars" : cuml_model.covars_,
                "weights": cuml_model.weights_,
                 "transmat": cuml_model.transmat_,
                 "startprob" : cuml_model.startprob_}

    to_csv(data_dict, path)