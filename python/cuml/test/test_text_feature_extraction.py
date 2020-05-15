#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
from cuml.feature_extraction.text import CountVectorizer
import cupy as cp
import pytest
from sklearn.feature_extraction.text import CountVectorizer as SkCountVect
from cudf import Series


def test_count_vectorizer():
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]

    res = CountVectorizer().fit_transform(Series(corpus))
    ref = SkCountVect().fit_transform(corpus)
    cp.testing.assert_array_equal(res, ref.toarray())


JUNK_FOOD_DOCS = (
    "the pizza pizza beer copyright",
    "the pizza burger beer copyright",
    "the the pizza beer beer copyright",
    "the burger beer beer copyright",
    "the coke burger coke copyright",
    "the coke burger burger",
)

NOTJUNK_FOOD_DOCS = (
    "the salad celeri copyright",
    "the salad salad sparkling water copyright",
    "the the celeri celeri copyright",
    "the tomato tomato salad water",
    "the tomato salad water copyright",
)

DOCS = JUNK_FOOD_DOCS + NOTJUNK_FOOD_DOCS
DOCS_GPU = Series(DOCS)


@pytest.mark.parametrize('ngram_range', [(1, 1), (1, 2)])
def test_word_analyzer(ngram_range):
    vec = CountVectorizer(ngram_range=ngram_range).fit(DOCS_GPU)
    ref = SkCountVect(ngram_range=ngram_range).fit(DOCS)
    assert ref.get_feature_names() == vec.get_feature_names().to_host()


def test_countvectorizer_custom_vocabulary():
    vocab = {"pizza": 0, "beer": 1}
    vocab_gpu = Series(vocab.keys())

    ref = SkCountVect(vocabulary=vocab).fit_transform(DOCS)
    X = CountVectorizer(vocabulary=vocab_gpu).fit_transform(DOCS_GPU)

    cp.testing.assert_array_equal(X, ref.toarray())


def test_countvectorizer_stop_words():
    ref = SkCountVect(stop_words='english').fit_transform(DOCS)
    X = CountVectorizer(stop_words='english').fit_transform(DOCS_GPU)
    cp.testing.assert_array_equal(X, ref.toarray())


def test_countvectorizer_empty_vocabulary():
    v = CountVectorizer(max_df=1.0, stop_words="english")
    # fit on stopwords only
    v.fit(Series(["to be or not to be", "and me too", "and so do you"]))
    assert False, "we shouldn't get here"
