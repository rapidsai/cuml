# Copyright (c) 2018, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cuml
from cuml.hmm.hmm_sampling import HMMSampler
import hmmlearn


from cuml.gmm.utils import *


class HMMTester :
    def __init__(self, kwargs):
        self.kwargs = kwargs

    def _reset(self):
        self.sk = hmmlearn.hmm.GMMHMM(**self.kwargs)
        self.cuml = cuml.hmm.HiddenMarkovModel(**self.kwargs)


    def test_workflow(self, X, lengths):
        self._reset()
        self.cuml._forward_backward(X, lengths, False, False)
        print(self.cuml.means_)

    def test_score_samples(self, X, lengths):
        cuml_out = self.cuml.score_samples(X, lengths)
        sk_out = self.sk.score_samples(X, lengths)
        return mae(cuml_out, sk_out)


if __name__ == '__main__':
    n_seq = 10
    n_dim = 3
    hmm_kwargs = {"n_components" : 5,
                  "n_mix" : 4,
                  "covariance_type" : "full"}

    hmm_sampler = HMMSampler(n_seq=n_seq,
                             n_dim=n_dim,
                             n_mix=hmm_kwargs["n_mix"],
                             n_components=hmm_kwargs["n_components"])
    X, lengths = hmm_sampler.sample_sequences()

    hmm_tester = HMMTester(hmm_kwargs)
    hmm_tester.test_workflow(X, lengths)