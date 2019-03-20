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


from cuml.hmm.utils.hmm_test_utils import *

#
# def test_gmmhmm():
#     n_seq = 10
#     n_dim = 3
#     hmm_kwargs = {"n_components": 5,
#                   "n_mix": 4,
#                   "covariance_type": "full"}
#     hmm_type = "gmm"
#
#     hmm_sampler = GMMHMMSampler(n_seq=n_seq,
#                                 n_dim=n_dim,
#                                 n_mix=hmm_kwargs["n_mix"],
#                                 n_components=hmm_kwargs["n_components"])
#     X, lengths = hmm_sampler.sample_sequences()
#
#     hmm_tester = GMMHMMTester(kwargs=hmm_kwargs, hmm_type=hmm_type)
#     hmm_tester.test_workflow(X, lengths)


def test_multinomialhmm():
    n_seq = 10
    n_components = 5
    n_features = 8

    hmm_sampler = MultinomialHMMSampler(n_seq=n_seq,
                                        n_components=n_components,
                                        n_features=n_features)
    X, lengths = hmm_sampler.sample_sequences()

    hmm_tester = MultinomialHMMTester(n_components=n_components,
                                      precision="double",
                                      random_state=None)
    hmm_tester.test_workflow(X, lengths)


if __name__ == '__main__':
    test_multinomialhmm()