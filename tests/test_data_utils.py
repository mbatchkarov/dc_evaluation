import os

import numpy as np
from eval.scripts.compress_labelled_data import jsonify_single_labelled_corpus

from eval.utils.data_utils import (get_tokenized_data,
                                   get_tokenizer_settings_from_conf,
                                   get_pipeline_fit_args)
from eval.utils.conf_file_utils import parse_config_file


def test_jsonify_XML_corpus():
    conf_file = 'tests/resources/conf/exp0/exp0.conf'
    conf, _ = parse_config_file(conf_file)
    train_set = conf['training_data']
    json_train_set = train_set + '.gz'
    tk = get_tokenizer_settings_from_conf(conf)

    # parse the XML directly
    x_tr, y_tr, _, _ = get_tokenized_data(train_set, tk)

    jsonify_single_labelled_corpus('unit_tests', train_set, conf_file)
    x_tr1, y_tr1, _, _ = get_tokenized_data(json_train_set, tk)

    # because the process of converting to json merges the train and test set, if a test set exists,
    # we need to merge them too in this test.
    for a, b in zip(x_tr, x_tr1):
        assert len(a[0]) == len(b) == 3
        assert set(str(f) for f in a[0].nodes()) == set(b)
    np.testing.assert_array_equal(y_tr, y_tr1)
    os.unlink(json_train_set)


def test_get_pipeline_fit_args():
    conf = {
        'feature_selection': {
            'must_be_in_thesaurus': True
        },
        'vectorizer': {
            'decode_token_handler': 'eval.pipeline.feature_handlers.SignifiedOnlyFeatureHandler',
            'random_neighbour_thesaurus': False,
        },
        'feature_extraction': {
            'train_time_opts': {},
            'decode_time_opts': {}
        },
        'vector_sources': {
            'neighbours_file': ['tests/resources/twos.vectors.txt'],
            'entries_of': 'tests/resources/ones.vectors.txt',
            'clusters_file': '',
            'is_thesaurus': False,
            'dummy_thesaurus': False,
        }
    }

    res = get_pipeline_fit_args(conf)
    v = res['vector_source']
    assert len(v) == 4
    assert set(v.keys()) == set('a/N b/V c/J d/N'.split())
    assert v.get_vector('a/N').A.ravel().tolist() == [1, 0, 0, 0]
    assert v.get_vector('c/J').A.ravel().tolist() == [0, 0, 1, 0]

    conf['vector_sources']['entries_of'] = conf['vector_sources']['neighbours_file'][0]
    print(conf)
    res = get_pipeline_fit_args(conf)
    v = res['vector_source']
    assert len(v) == 8
    assert set(v.keys()) == set('a/N b/V c/J d/N e/N f/V g/J h/N'.split())
    assert v.get_vector('a/N').A.ravel().tolist() == [1, 0, 0, 0]
    assert v.get_vector('c/J').A.ravel().tolist() == [0, 0, 1, 0]
    assert v.get_vector('g/J').A.ravel().tolist() == [0, 0, 1.1, 0]
