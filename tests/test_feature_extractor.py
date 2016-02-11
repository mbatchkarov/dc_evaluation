import pytest
import networkx as nx
from discoutils.tokens import DocumentFeature, Token

from eval.pipeline.feature_extractors import FeatureExtractor
from eval.pipeline.tokenizers import XmlTokenizer


@pytest.fixture
def valid_AN_features():
    return [('big', 'cat'), ('black', 'cat'), ('small', 'bird'), ('red', 'bird')]


@pytest.fixture
def extractor():
    return FeatureExtractor(extract_unigram_features='J,N',
                            extract_phrase_features=['AN', 'NN', 'VO', 'SVO'])


@pytest.fixture
def black_cat_parse_tree():
    with open('tests/resources/tokenizer/black_cat.tagged') as infile:
        txt = infile.read()
    t = XmlTokenizer()
    parse_tree = t.tokenize_doc(txt)[0]
    assert len(parse_tree.nodes()) == 14  # 15 tokens, 1 on of which is punctuation
    return parse_tree


def test_extract_features_from_correct_dependency_tree(black_cat_parse_tree, extractor, valid_AN_features):
    features = extractor.extract_features_from_single_dependency_tree(black_cat_parse_tree)

    for adj, noun in valid_AN_features:
        f = DocumentFeature('AN', (Token(adj, 'J'), Token(noun, 'N')))
        assert f in features

    assert DocumentFeature('VO', (Token('eat', 'V'), Token('bird', 'N'))) in features
    assert DocumentFeature('SVO', (Token('cat', 'N'), Token('eat', 'V'), Token('bird', 'N'))) \
           in features

    assert DocumentFeature('NN', (Token('heart', 'N'), Token('surgery', 'N'))) in features


def test_extract_features_with_disabled_features(black_cat_parse_tree, extractor, valid_AN_features):
    extractor.extract_phrase_features = ['AN', 'NN']

    features = extractor.extract_features_from_single_dependency_tree(black_cat_parse_tree)

    for adj, noun in valid_AN_features:
        f = DocumentFeature('AN', (Token(adj, 'J'), Token(noun, 'N')))
        assert f in features

    assert DocumentFeature('VO', (Token('eat', 'V'), Token('bird', 'N'))) not in features
    assert DocumentFeature('SVO', (Token('cat', 'N'), Token('eat', 'V'), Token('bird', 'N'))) \
           not in features

    assert DocumentFeature('NN', (Token('heart', 'N'), Token('surgery', 'N'))) in features


def test_extract_features_from_empty_dependency_tree(extractor):
    features = extractor.extract_features_from_single_dependency_tree(nx.DiGraph())
    assert not features

    features = extractor.extract_features_from_tree_list([nx.DiGraph()])
    assert not features


def test_remove_features_containing_named_entities(extractor, black_cat_parse_tree):
    features = extractor.extract_features_from_single_dependency_tree(black_cat_parse_tree)

    cleaned_features = extractor.remove_features_containing_named_entities(features)
    assert cleaned_features == features

    # make the token cat/N into a named entity
    features[0].tokens[1].ner = 'PERSON'
    cleaned_features = extractor.remove_features_containing_named_entities(features)
    assert len(cleaned_features) == len(features) - 4  # 4 features contain the Token 'cat/N'


@pytest.mark.parametrize(
    ('change_to', 'expected_feature_count'),
    [
        ('amod', 4),
        ('nsubj', 0),
        ('dobj', 4)
    ]
)
def test_extract_features_from_dependency_tree_with_wrong_relation_types(black_cat_parse_tree,
                                                                         extractor,
                                                                         change_to,
                                                                         expected_feature_count,
                                                                         valid_AN_features):
    # change all dependencies to amod
    for source, target, data in black_cat_parse_tree.edges(data=True):
        if data['type'] != change_to:
            data['type'] = change_to

    for source, target, data in black_cat_parse_tree.edges(data=True):
        print(data)

    features = extractor.extract_features_from_single_dependency_tree(black_cat_parse_tree)
    # if all relations are changed to AMOD, we should get two adjective per noun
    # if all relations are changed to NSUBJ, we should get no AN/VO/SVO features as we're missing the required relations
    # to build and AN feature, we need an AMOD relation between a J and a N
    # to build and VO feature, we need an DOBJ relation between a V and a N
    # to build and AN feature, we need an DOBJ relation between a V and a N and a NSUBJ between the same V and another N
    assert len(features) == expected_feature_count

    # check that only amods between adj and nouns are extracted
    for adj, noun in valid_AN_features:
        f = DocumentFeature('AN', (Token(adj, 'J'), Token(noun, 'N')))
        assert (f in features) == (change_to == 'amod')

    # check that no VO/SVO features are extracted
    feature = DocumentFeature('VO', (Token('eat', 'V'), Token('bird', 'N')))
    assert (feature in features) == (change_to == 'dobj')

    feature = DocumentFeature('SVO', (Token('cat', 'N'), Token('eat', 'V'), Token('bird', 'N')))
    assert feature not in features  # we're missing the required relations to build SVO
