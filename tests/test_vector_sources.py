import pytest
from numpy.testing import assert_array_equal
from discoutils.thesaurus_loader import Vectors
from discoutils.tokens import DocumentFeature, Token

DIM = 10

unigram_feature = DocumentFeature('1-GRAM', (Token('a', 'N'),))
unk_unigram_feature = DocumentFeature('1-GRAM', ((Token('unk', 'UNK')),))
bigram_feature = DocumentFeature('2-GRAM', (Token('a', 'N'), Token('b', 'V')))
unk_bigram_feature = DocumentFeature('2-GRAM', (Token('a', 'N'), Token('UNK', 'UNK')))
an_feature = DocumentFeature('AN', (Token('c', 'J'), Token('a', 'n')))
known_features = set([unigram_feature, bigram_feature, an_feature])
all_features = set([unigram_feature, bigram_feature, an_feature, unk_unigram_feature, unk_bigram_feature])


@pytest.fixture(scope='module')
def small_vectors():
    return Vectors.from_tsv('eval/resources/thesauri/small.txt.events.strings')


@pytest.fixture
def vectors_a():
    return Vectors.from_tsv('eval/resources/exp0-0a.strings')


@pytest.fixture
def ones_vectors():
    return Vectors.from_tsv('eval/resources/ones.vectors.txt')


@pytest.fixture
def ones_vectors_no_pos():
    return Vectors.from_tsv('eval/resources/ones.vectors.nopos.txt',
                            enforce_word_entry_pos_format=False)


def test_unigr_source_get_vector(small_vectors):
    # vectors come out right
    # a/N	amod:c	2   T:t1	1	T:t2	2	T:t3	3
    assert_array_equal(
        small_vectors.get_vector('a/N').todense(),
        [[0., 2., 0., 1., 2., 3., 0., 0.]]
    )

    # vocab is in sorted order
    assert ['also/RB', 'amod:c', 'now/RB', 't:t1', 't:t2', 't:t3', 't:t4', 't:t5', ] == small_vectors.columns

    assert small_vectors.get_vector('jfhjgjdfyjhgb/N') is None
    assert small_vectors.get_vector('jfhjgjdfyjhgb/J') is None


def test_unigr_source_contains(small_vectors):
    """
    Test if the unigram model only accepts unigram features
    """
    # for thing in (known_features | unk_unigram_feature | unk_bigram_feature):
    assert str(unigram_feature) in small_vectors
    for thing in (unk_unigram_feature, bigram_feature, unk_unigram_feature):
        assert str(thing) not in small_vectors
