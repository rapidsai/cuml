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
    # fitting only on stopwords will result in an empty vocabulary
    with pytest.raises(ValueError):
        v.fit(Series(["to be or not to be", "and me too", "and so do you"]))


def test_countvectorizer_max_features():
    expected_vocabulary = {'burger', 'beer', 'salad', 'pizza'}
    expected_stop_words = {'celeri', 'tomato', 'copyright', 'coke',
                           'sparkling', 'water', 'the'}

    # test bounded number of extracted features
    vec = CountVectorizer(max_df=0.6, max_features=4)
    vec.fit(DOCS_GPU)
    assert set(vec.get_feature_names().to_host()) == expected_vocabulary
    assert set(vec.stop_words_.to_host()) == expected_stop_words


def test_countvectorizer_max_features_counts():
    JUNK_FOOD_DOCS_GPU = Series(JUNK_FOOD_DOCS)

    cv_1 = CountVectorizer(max_features=1)
    cv_3 = CountVectorizer(max_features=3)
    cv_None = CountVectorizer(max_features=None)

    counts_1 = cv_1.fit_transform(JUNK_FOOD_DOCS_GPU).sum(axis=0)
    counts_3 = cv_3.fit_transform(JUNK_FOOD_DOCS_GPU).sum(axis=0)
    counts_None = cv_None.fit_transform(JUNK_FOOD_DOCS_GPU).sum(axis=0)

    features_1 = cv_1.get_feature_names()
    features_3 = cv_3.get_feature_names()
    features_None = cv_None.get_feature_names()

    # The most common feature is "the", with frequency 7.
    assert 7 == counts_1.max()
    assert 7 == counts_3.max()
    assert 7 == counts_None.max()

    # The most common feature should be the same
    def as_index(x): return x.astype(cp.int32).get()
    assert ["the"] == features_1[as_index(cp.argmax(counts_1))].to_host()
    assert ["the"] == features_3[as_index(cp.argmax(counts_3))].to_host()
    assert ["the"] == features_None[as_index(cp.argmax(counts_None))].to_host()


def test_countvectorizer_max_df():
    test_data = Series(['abc', 'dea', 'eat'])
    vect = CountVectorizer(analyzer='char', max_df=1.0)
    vect.fit(test_data)
    assert 'a' in vect.vocabulary_.to_host()
    assert len(vect.vocabulary_.to_host()) == 6
    assert len(vect.stop_words_) == 0

    vect.max_df = 0.5  # 0.5 * 3 documents -> max_doc_count == 1.5
    vect.fit(test_data)
    assert 'a' not in vect.vocabulary_.to_host()  # {ae} ignored
    assert len(vect.vocabulary_.to_host()) == 4    # {bcdt} remain
    assert 'a' in vect.stop_words_.to_host()
    assert len(vect.stop_words_) == 2

    vect.max_df = 1
    vect.fit(test_data)
    assert 'a' not in vect.vocabulary_.to_host()  # {ae} ignored
    assert len(vect.vocabulary_.to_host()) == 4    # {bcdt} remain
    assert 'a' in vect.stop_words_.to_host()
    assert len(vect.stop_words_) == 2
