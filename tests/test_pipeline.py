# coding=utf-8
import glob
import os

import pytest
import numpy as np
from numpy.ma import std
import numpy.testing as t
import scipy.sparse as sp

from eval.pipeline.thesauri import DummyThesaurus
from eval import evaluate
from eval.scripts.compress_labelled_data import jsonify_single_labelled_corpus
from eval.utils.conf_file_utils import parse_config_file
from tests.test_feature_selectors import strip
from eval.utils.data_utils import get_tokenized_data

tsv_file = 'tests/resources/exp0-0b.strings'
tokenizer_opts = {
    'normalise_entities': False,
    'use_pos': True,
    'coarse_pos': True,
    'lemmatize': True,
    'lowercase': True,
    'remove_stopwords': False,
    'remove_short_words': False
}

training_matrix = np.array([
    [1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1],
])

pruned_training_matrix = np.array([
    [1, 1, 0],
    [0, 0, 1],
])
pruned_vocab = {'a/N': 0, 'b/N': 1, 'd/N': 2}
full_vocab = {'a/N': 0, 'b/N': 1, 'c/N': 2, 'd/N': 3, 'e/N': 4, 'f/N': 5}


def teardown_module(module):
    """
    This is a pytest module-level teardown function
    :param module:
    """
    for pattern in ['PostVectDump_test_main*', 'stats-test_main-cv12345*']:
        files = glob.glob(pattern)
        for f in files:
            if os.path.exists(f):
                os.remove(f)


@pytest.fixture(params=['xml', 'json'])
def data(request):
    """
    Returns path to a labelled dataset on disk
    """
    kind = request.param
    prefix = 'tests/resources/test-baseline'
    tr_path = '%s-tr' % prefix
    ev_path = '%s-ev' % prefix

    if kind == 'xml':
        # return the raw corpus in XML
        return tr_path, ev_path
    if kind == 'json':
        # convert corpus to gzipped JSON and try again
        jsonify_single_labelled_corpus('unit_tests', tr_path, tokenizer_conf=tokenizer_opts)
        jsonify_single_labelled_corpus('unit_tests', ev_path, tokenizer_conf=tokenizer_opts)
        return tr_path + '.gz', ev_path + '.gz'


@pytest.fixture
def conf(tmpdir):
    # load default configuration
    tmpfile = tmpdir.join('blank')
    with open(str(tmpfile), 'w'):
        pass  # touch
    res, _ = parse_config_file(str(tmpfile), confrc='conf/confrc', quit_on_error=False)

    res['feature_extraction'].update({
        'class': 'eval.pipeline.bov.ThesaurusVectorizer',
        'min_df': 1,
        'k': 10,  # use all thesaurus entries
        'train_token_handler': 'eval.pipeline.feature_handlers.BaseFeatureHandler',
        'decode_token_handler': 'eval.pipeline.feature_handlers.BaseFeatureHandler',
        'random_neighbour_thesaurus': False,
        'train_time_opts': dict(extract_unigram_features=['J', 'N', 'V'],
                                extract_phrase_features=[]),
        'decode_time_opts': dict(extract_unigram_features=['J', 'N', 'V'],
                                 extract_phrase_features=[])
    })

    res['feature_selection'].update({
        'run': True,
        'method': 'eval.pipeline.feature_selectors.VectorBackedSelectKBest',
        'scoring_function': 'sklearn.feature_selection.chi2',
        'must_be_in_thesaurus': False,
        'k': 'all',
    })

    res['vector_sources']['is_thesaurus'] = True
    return res


def _vectorize_data(data_paths, config, dummy=False):
    if dummy:
        config['vector_sources']['dummy_thesaurus'] = True
        config['vector_sources']['neighbours_file'] = []
    else:
        config['vector_sources']['neighbours_file'] = [tsv_file]

    config['vector_sources']['neighbour_strategy'] = 'linear'
    config['name'] = 'test_main'
    config['debug_level'] = 2
    config['output_dir'] = '.'
    pipeline, fit_params = evaluate._build_pipeline(config, 12345)

    x_tr, y_tr, x_test, y_test = get_tokenized_data(data_paths[0], tokenizer_opts, test_data=data_paths[1])

    x1 = pipeline.fit_transform(x_tr, y_tr, **fit_params)
    if 'fs' in pipeline.named_steps:
        pipeline.named_steps['vect'].vocabulary_ = pipeline.named_steps['fs'].vocabulary_

    voc = pipeline.named_steps['fs'].vocabulary_
    x2 = pipeline.transform(x_test)

    return x1, x2, voc


def test_nondistributional_baseline_without_feature_selection(data, conf):
    x1, x2, voc = _vectorize_data(data, conf)
    assert full_vocab == strip(voc)

    assert isinstance(x1, sp.spmatrix)
    t.assert_array_equal(
        x1.toarray(),
        training_matrix
    )

    t.assert_array_equal(
        x2.toarray(),
        np.array(
            [
                [4, 1, 2, 0, 0, 0],
            ]
        )
    )


def test_baseline_use_all_features_with__signifier_signified(data, conf):
    conf['feature_selection']['must_be_in_thesaurus'] = False
    conf['vectorizer']['decode_token_handler'] = \
        'eval.pipeline.feature_handlers.SignifierSignifiedFeatureHandler'
    conf['vectorizer']['k'] = 1

    x1, x2, voc = _vectorize_data(data, conf)

    assert full_vocab == strip(voc)

    assert isinstance(x1, sp.spmatrix)
    t.assert_array_equal(
        x1.toarray(),
        training_matrix
    )

    t.assert_array_almost_equal(
        x2.toarray(),
        np.array(
            [
                [4, 1, 2, 2.1, 0, 0]
            ]
        )
    )


def test_baseline_ignore_nonthesaurus_features_with_signifier_signified(data, conf):
    conf['feature_selection']['must_be_in_thesaurus'] = True
    conf['vectorizer']['decode_token_handler'] = \
        'eval.pipeline.feature_handlers.SignifierSignifiedFeatureHandler'
    conf['vectorizer']['k'] = 1

    x1, x2, voc = _vectorize_data(data, conf)
    assert pruned_vocab == strip(voc)
    assert isinstance(x1, sp.spmatrix)
    t.assert_array_equal(
        x1.toarray(),
        pruned_training_matrix
    )

    t.assert_array_almost_equal(
        x2.toarray(),
        np.array(
            [
                [4, 1, 2.1]
            ]
        )
    )


def test_baseline_use_all_features_with_signified(data, conf):
    conf['feature_selection']['must_be_in_thesaurus'] = False
    conf['vectorizer']['decode_token_handler'] = \
        'eval.pipeline.feature_handlers.SignifiedOnlyFeatureHandler'
    conf['vectorizer']['k'] = 1  # equivalent to max

    x1, x2, voc = _vectorize_data(data, conf)

    assert full_vocab == strip(voc)

    assert isinstance(x1, sp.spmatrix)
    t.assert_array_equal(
        x1.toarray(),
        training_matrix
    )

    t.assert_array_almost_equal(
        x2.toarray(),
        np.array(
            [
                [0, 0, 0, 4.4, 0, 0],
            ]
        )
    )


def test_baseline_ignore_nonthesaurus_features_with_signified(data, conf):
    conf['feature_selection']['must_be_in_thesaurus'] = True
    conf['vectorizer']['decode_token_handler'] = \
        'eval.pipeline.feature_handlers.SignifiedOnlyFeatureHandler'
    conf['vectorizer']['k'] = 1

    x1, x2, voc = _vectorize_data(data, conf)

    assert pruned_vocab == strip(voc)

    assert isinstance(x1, sp.spmatrix)
    t.assert_array_equal(
        x1.toarray(),
        pruned_training_matrix
    )

    t.assert_array_almost_equal(
        x2.toarray(),
        np.array(
            [
                [0, 0, 4.4]
            ]
        )
    )


def test_baseline_use_all_features_with_signified_random(data, conf):
    conf['feature_selection']['must_be_in_thesaurus'] = False
    conf['vectorizer']['decode_token_handler'] = \
        'eval.pipeline.feature_handlers.SignifiedOnlyFeatureHandler'
    conf['vectorizer']['k'] = 1

    x1, x2, voc = _vectorize_data(data, conf, dummy=True)

    assert full_vocab == strip(voc)

    assert isinstance(x1, sp.spmatrix)
    t.assert_array_equal(
        x1.toarray(),
        training_matrix
    )

    t.assert_array_almost_equal(
        x2.toarray(),
        np.array(
            [
                [0, 11.0, 0, 0, 0, 0],
            ]
        )
    )
    # the thesaurus will always say the neighbour for something is
    # b/N with a similarity of 1, and we look up 11 tokens overall in
    # the test document
    x1, x2, voc = _vectorize_data(data, conf, dummy=True)
    assert x2.sum(), 11.0
    assert std(x2.todense()) > 0
    # seven tokens will be looked up, with random in-vocabulary neighbours
    # returned each time. Std>0 shows that it's not the same thing
    # returned each time
    # print x2
