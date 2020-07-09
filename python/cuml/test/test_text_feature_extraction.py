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
from cuml.feature_extraction.text import TfidfVectorizer
import cupy as cp
import pytest
from sklearn.feature_extraction.text import CountVectorizer as SkCountVect
from sklearn.feature_extraction.text import TfidfVectorizer as SkTfidfVect
from cudf import Series
from numpy.testing import assert_array_equal
import numpy as np


def test_count_vectorizer():
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]

    res = CountVectorizer().fit_transform(Series(corpus))
    ref = SkCountVect().fit_transform(corpus)
    cp.testing.assert_array_equal(res.todense(), ref.toarray())


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

EMPTY_DOCS = ("",)

DOCS = JUNK_FOOD_DOCS + EMPTY_DOCS + NOTJUNK_FOOD_DOCS + EMPTY_DOCS
DOCS_GPU = Series(DOCS)

NGRAM_RANGES = [(1, 1), (1, 2), (2, 3)]
NGRAM_IDS = [f'ngram_range={str(r)}' for r in NGRAM_RANGES]


@pytest.mark.parametrize('ngram_range', NGRAM_RANGES, ids=NGRAM_IDS)
def test_word_analyzer(ngram_range):
    vec = CountVectorizer(ngram_range=ngram_range).fit(DOCS_GPU)
    ref = SkCountVect(ngram_range=ngram_range).fit(DOCS)
    assert ref.get_feature_names() == vec.get_feature_names().tolist()


def test_countvectorizer_custom_vocabulary():
    vocab = {"pizza": 0, "beer": 1}
    vocab_gpu = Series(vocab.keys())

    ref = SkCountVect(vocabulary=vocab).fit_transform(DOCS)
    X = CountVectorizer(vocabulary=vocab_gpu).fit_transform(DOCS_GPU)
    cp.testing.assert_array_equal(X.todense(), ref.toarray())


def test_countvectorizer_stop_words():
    ref = SkCountVect(stop_words='english').fit_transform(DOCS)
    X = CountVectorizer(stop_words='english').fit_transform(DOCS_GPU)
    cp.testing.assert_array_equal(X.todense(), ref.toarray())


def test_countvectorizer_empty_vocabulary():
    v = CountVectorizer(max_df=1.0, stop_words="english")
    # fitting only on stopwords will result in an empty vocabulary
    with pytest.raises(ValueError):
        v.fit(Series(["to be or not to be", "and me too", "and so do you"]))


def test_countvectorizer_stop_words_ngrams():
    stop_words_doc = Series(["and me too andy andy too"])
    expected_vocabulary = ["andy andy"]

    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english')
    vec.fit(stop_words_doc)

    assert expected_vocabulary == vec.get_feature_names().tolist()


def test_countvectorizer_max_features():
    expected_vocabulary = {'burger', 'beer', 'salad', 'pizza'}
    expected_stop_words = {'celeri', 'tomato', 'copyright', 'coke',
                           'sparkling', 'water', 'the'}

    # test bounded number of extracted features
    vec = CountVectorizer(max_df=0.6, max_features=4)
    vec.fit(DOCS_GPU)
    assert set(vec.get_feature_names().tolist()) == expected_vocabulary
    assert set(vec.stop_words_.tolist()) == expected_stop_words


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
    def as_index(x): return x.astype(cp.int32).item()
    assert "the" == features_1[as_index(cp.argmax(counts_1))]
    assert "the" == features_3[as_index(cp.argmax(counts_3))]
    assert "the" == features_None[as_index(cp.argmax(counts_None))]


def test_countvectorizer_max_df():
    test_data = Series(['abc', 'dea', 'eat'])
    vect = CountVectorizer(analyzer='char', max_df=1.0)
    vect.fit(test_data)
    assert 'a' in vect.vocabulary_.tolist()
    assert len(vect.vocabulary_.tolist()) == 6
    assert len(vect.stop_words_) == 0

    vect.max_df = 0.5  # 0.5 * 3 documents -> max_doc_count == 1.5
    vect.fit(test_data)
    assert 'a' not in vect.vocabulary_.tolist()  # {ae} ignored
    assert len(vect.vocabulary_.tolist()) == 4    # {bcdt} remain
    assert 'a' in vect.stop_words_.tolist()
    assert len(vect.stop_words_) == 2

    vect.max_df = 1
    vect.fit(test_data)
    assert 'a' not in vect.vocabulary_.tolist()  # {ae} ignored
    assert len(vect.vocabulary_.tolist()) == 4    # {bcdt} remain
    assert 'a' in vect.stop_words_.tolist()
    assert len(vect.stop_words_) == 2


def test_vectorizer_min_df():
    test_data = Series(['abc', 'dea', 'eat'])
    vect = CountVectorizer(analyzer='char', min_df=1)
    vect.fit(test_data)
    assert 'a' in vect.vocabulary_.tolist()
    assert len(vect.vocabulary_.tolist()) == 6
    assert len(vect.stop_words_) == 0

    vect.min_df = 2
    vect.fit(test_data)
    assert 'c' not in vect.vocabulary_.tolist()  # {bcdt} ignored
    assert len(vect.vocabulary_.tolist()) == 2    # {ae} remain
    assert 'c' in vect.stop_words_.tolist()
    assert len(vect.stop_words_) == 4

    vect.min_df = 0.8  # 0.8 * 3 documents -> min_doc_count == 2.4
    vect.fit(test_data)
    assert 'c' not in vect.vocabulary_.tolist()  # {bcdet} ignored
    assert len(vect.vocabulary_.tolist()) == 1    # {a} remains
    assert 'c' in vect.stop_words_.tolist()
    assert len(vect.stop_words_) == 5


def test_count_binary_occurrences():
    # by default multiple occurrences are counted as longs
    test_data = Series(['aaabc', 'abbde'])
    vect = CountVectorizer(analyzer='char', max_df=1.0)
    X = cp.asnumpy(vect.fit_transform(test_data).todense())
    assert_array_equal(['a', 'b', 'c', 'd', 'e'],
                       vect.get_feature_names().tolist())
    assert_array_equal([[3, 1, 1, 0, 0],
                        [1, 2, 0, 1, 1]], X)

    # using boolean features, we can fetch the binary occurrence info
    # instead.
    vect = CountVectorizer(analyzer='char', max_df=1.0, binary=True)
    X = cp.asnumpy(vect.fit_transform(test_data).todense())
    assert_array_equal([[1, 1, 1, 0, 0],
                        [1, 1, 0, 1, 1]], X)

    # check the ability to change the dtype
    vect = CountVectorizer(analyzer='char', max_df=1.0,
                           binary=True, dtype=cp.float32)
    X = vect.fit_transform(test_data)
    assert X.dtype == cp.float32


def test_vectorizer_inverse_transform():
    vectorizer = CountVectorizer()
    transformed_data = vectorizer.fit_transform(DOCS_GPU)
    inversed_data = vectorizer.inverse_transform(transformed_data)

    sk_vectorizer = SkCountVect()
    sk_transformed_data = sk_vectorizer.fit_transform(DOCS)
    sk_inversed_data = sk_vectorizer.inverse_transform(sk_transformed_data)

    for doc, sk_doc in zip(inversed_data, sk_inversed_data):
        doc = np.sort(doc.tolist())
        sk_doc = np.sort(sk_doc)
        if len(doc) + len(sk_doc) == 0:
            continue
        assert_array_equal(doc, sk_doc)


@pytest.mark.parametrize('ngram_range', NGRAM_RANGES, ids=NGRAM_IDS)
def test_space_ngrams(ngram_range):
    data = ['abc      def. 123 456    789']
    data_gpu = Series(data)
    vec = CountVectorizer(ngram_range=ngram_range).fit(data_gpu)
    ref = SkCountVect(ngram_range=ngram_range).fit(data)
    assert ref.get_feature_names() == vec.get_feature_names().tolist()


def test_empty_doc_after_limit_features():
    data = ['abc abc def',
            'def abc',
            'ghi']
    data_gpu = Series(data)
    count = CountVectorizer(min_df=2).fit_transform(data_gpu)
    ref = SkCountVect(min_df=2).fit_transform(data)
    cp.testing.assert_array_equal(count.todense(), ref.toarray())


def test_countvectorizer_separate_fit_transform():
    res = CountVectorizer().fit(DOCS_GPU).transform(DOCS_GPU)
    ref = SkCountVect().fit(DOCS).transform(DOCS)
    cp.testing.assert_array_equal(res.todense(), ref.toarray())


def test_non_ascii():
    non_ascii = ('This is ascii,',
                 'but not this Αγγλικά.')
    non_ascii_gpu = Series(non_ascii)

    cv = CountVectorizer()
    res = cv.fit_transform(non_ascii_gpu)
    ref = SkCountVect().fit_transform(non_ascii)

    assert 'αγγλικά' in set(cv.get_feature_names().tolist())
    cp.testing.assert_array_equal(res.todense(), ref.toarray())


def test_only_delimiters():
    data = ['abc def. 123',
            '   ',
            '456 789']
    data_gpu = Series(data)
    res = CountVectorizer().fit_transform(data_gpu)
    ref = SkCountVect().fit_transform(data)
    cp.testing.assert_array_equal(res.todense(), ref.toarray())


@pytest.mark.parametrize('analyzer', ['char', 'char_wb'])
@pytest.mark.parametrize('ngram_range', NGRAM_RANGES, ids=NGRAM_IDS)
def test_character_ngrams(analyzer, ngram_range):
    data = ['ab c',
            ''
            'edf gh']

    res = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    res.fit(Series(data))

    ref = SkCountVect(analyzer=analyzer, ngram_range=ngram_range).fit(data)

    assert ref.get_feature_names() == res.get_feature_names().tolist()


@pytest.mark.parametrize('query', [Series(['science aa', '', 'a aa aaa']),
                                   Series(['science aa', '']),
                                   Series(['science'])])
def test_transform_unsigned_categories(query):
    token = 'a'
    thousand_tokens = list()
    for i in range(1000):
        thousand_tokens.append(token)
        token += 'a'
    thousand_tokens[128] = 'science'

    vec = CountVectorizer().fit(Series(thousand_tokens))
    res = vec.transform(query)

    assert res.shape[0] == len(query)


# ----------------------------------------------------------------
# TfidfVectorizer tests are already covered by CountVectorizer and
# TfidfTransformer so we only do the bare minimum tests here
# ----------------------------------------------------------------

def test_tfidf_vectorizer_setters():
    tv = TfidfVectorizer(norm='l2', use_idf=False, smooth_idf=False,
                         sublinear_tf=False)
    tv.norm = 'l1'
    assert tv._tfidf.norm == 'l1'
    tv.use_idf = True
    assert tv._tfidf.use_idf
    tv.smooth_idf = True
    assert tv._tfidf.smooth_idf
    tv.sublinear_tf = True
    assert tv._tfidf.sublinear_tf


def test_tfidf_vectorizer_idf_setter():
    orig = TfidfVectorizer(use_idf=True)
    orig.fit(DOCS_GPU)
    copy = TfidfVectorizer(vocabulary=orig.vocabulary_, use_idf=True)
    copy.idf_ = orig.idf_[0]
    cp.testing.assert_array_almost_equal(copy.transform(DOCS_GPU).todense(),
                                         orig.transform(DOCS_GPU).todense())


@pytest.mark.parametrize('norm', ['l1', 'l2', None])
@pytest.mark.parametrize('use_idf', [True, False])
@pytest.mark.parametrize('smooth_idf', [True, False])
@pytest.mark.parametrize('sublinear_tf', [True, False])
def test_tfidf_vectorizer(norm, use_idf, smooth_idf, sublinear_tf):
    tfidf_mat = TfidfVectorizer(
        norm=norm, use_idf=use_idf,
        smooth_idf=smooth_idf, sublinear_tf=sublinear_tf
    ).fit_transform(DOCS_GPU)

    ref = SkTfidfVect(
        norm=norm, use_idf=use_idf,
        smooth_idf=smooth_idf, sublinear_tf=sublinear_tf
    ).fit_transform(DOCS)

    cp.testing.assert_array_almost_equal(tfidf_mat.todense(), ref.toarray())
