from glob import glob
import logging
import os
import random

import numpy as np
import pandas as pd
from sklearn.datasets import load_files

from discoutils.thesaurus_loader import Vectors, Thesaurus
from discoutils.misc import is_gzipped, _check_file_magic
from eval.pipeline.feature_extractors import FeatureExtractor
from eval.pipeline.tokenizers import GzippedJsonTokenizer, ConllTokenizer, XmlTokenizer
from eval.utils.conf_file_utils import parse_config_file
from eval.pipeline.thesauri import RandomThesaurus, DummyThesaurus
from eval.pipeline.multivectors import MultiVectors


def tokenize_data(data, tokenizer, corpus_ids):
    """
    :param data: list of strings, contents of documents to tokenize
    :param tokenizer:
    :param corpus_ids:  list-like, names of the training corpus (and optional testing corpus), used for
    retrieving pre-tokenized data from joblib cache
    """
    x_tr, y_tr, x_test, y_test = data
    # todo this logic needs to be moved to feature extractor
    x_tr = tokenizer.tokenize_corpus(x_tr, corpus_ids[0])
    if x_test is not None and y_test is not None and corpus_ids[1] is not None:
        x_test = tokenizer.tokenize_corpus(x_test, corpus_ids[1])
    return x_tr, y_tr, x_test, y_test


def load_text_data_into_memory(training_path, test_path=None, shuffle_targets=False):
    x_train, y_train = _get_data_iterators(training_path, shuffle_targets=shuffle_targets)

    if test_path:
        logging.info('Loading raw test set %s' % test_path)
        x_test, y_test = _get_data_iterators(test_path, shuffle_targets=shuffle_targets)
    else:
        x_test, y_test = None, None
    return (x_train, y_train, x_test, y_test), (training_path, test_path)


def get_tokenizer_settings_from_conf(conf):
    return {'normalise_entities': conf['feature_extraction']['normalise_entities'],
            'use_pos': conf['feature_extraction']['use_pos'],
            'coarse_pos': conf['feature_extraction']['coarse_pos'],
            'lemmatize': conf['feature_extraction']['lemmatize'],
            'lowercase': conf['tokenizer']['lowercase'],
            'remove_stopwords': conf['tokenizer']['remove_stopwords'],
            'remove_short_words': conf['tokenizer']['remove_short_words'],
            'remove_long_words': conf['tokenizer']['remove_long_words']}


def get_tokenizer_settings_from_conf_file(conf_file):
    conf, _ = parse_config_file(conf_file)
    return get_tokenizer_settings_from_conf(conf)


def get_tokenized_data(training_path, tokenizer_conf, shuffle_targets=False,
                       test_data='', *args, **kwargs):
    """
    Loads data from either XML or compressed JSON
    :param gzip_json: set to True of False to force this method to read XML/JSON. Otherwise the type of
     input data is determined by the presence or absence of a .gz extension on the training_path
    :param args:
    """
    if is_gzipped(training_path):
        tokenizer = GzippedJsonTokenizer(**tokenizer_conf)
        x_tr, y_tr = tokenizer.tokenize_corpus(training_path)
        if test_data:
            x_test, y_test = tokenizer.tokenize_corpus(test_data)
        else:
            x_test, y_test = None, None
        return x_tr, np.array(y_tr), x_test, np.array(y_test) if y_test else y_test
    # todo maybe we want to keep the XML code. In that case check if the file is XML or CoNLL

    # m011303:sample-data mmb28$ file aclImdb-tagged/neg/train_neg_12490_1.txt.tagged
    # aclImdb-tagged/neg/train_neg_12490_1.txt.tagged: XML  document text
    elif _check_file_magic(training_path, b'directory'):
        # need to find out if these are XML or CoNLL files
        a_file = glob(os.path.join(training_path, '*', '*'))[0]
        if _check_file_magic(a_file, b'XML'):
            tokenizer = XmlTokenizer(**tokenizer_conf)
        else:
            # must be ConLL then
            tokenizer = ConllTokenizer(**tokenizer_conf)
        raw_data, data_ids = load_text_data_into_memory(training_path=training_path,
                                                        test_path=test_data,
                                                        shuffle_targets=shuffle_targets)
        return tokenize_data(raw_data, tokenizer, data_ids)
    else:
        raise ValueError('Input is neither a gzipped file containing all data nor a directory')


def _get_data_iterators(path, shuffle_targets=False):
    """
    Returns iterators over the text of the data.

    :param path: The source folder to be read. Should contain data in the
     mallet format.
    :param shuffle_targets: If true, the true labels of the data set will be shuffled. This is useful as a
    sanity check
    """

    logging.info('Using a file content generator with source %(path)s' % locals())
    if not os.path.isdir(path):
        raise ValueError('The provided source path %s has to be a directory containing data in the mallet format'
                         ' (class per directory, document per file).' % path)

    dataset = load_files(path, shuffle=False, load_content=False)
    logging.info('Targets are: %s', dataset.target_names)
    data_iterable = dataset.filenames
    if shuffle_targets:
        logging.warning('RANDOMIZING TARGETS')
        random.shuffle(dataset.target)

    return data_iterable, np.array(dataset.target_names)[dataset.target]


def get_pipeline_fit_args(conf):
    """
    Builds a dict of resources that document vectorizers require at fit time. These currently include
    various kinds of distributional information, e.g. word vectors or cluster ID for words and phrases.
    Example:
    {'vector_source': <DenseVectors object>} or {'clusters': <pd.DataFrame of word clusters>}
    :param conf: configuration dict
    :raise ValueError: if the conf is wrong in any way
    """
    result = dict()
    train_time_extractor = FeatureExtractor().update(**conf['feature_extraction']).\
        update(**conf['feature_extraction']['train_time_opts'])
    result['train_time_extractor'] = train_time_extractor
    decode_time_extractor = FeatureExtractor().update(**conf['feature_extraction']).\
        update(**conf['feature_extraction']['decode_time_opts'])
    result['decode_time_extractor'] = decode_time_extractor


    vectors_exist = conf['feature_selection']['must_be_in_thesaurus']
    handler_ = conf['vectorizer']['decode_token_handler']
    random_thes = conf['vectorizer']['random_neighbour_thesaurus']
    dummy_thes = conf['vector_sources']['dummy_thesaurus']
    vs_params = conf['vector_sources']
    vectors_path = vs_params['neighbours_file']
    clusters_path = vs_params['clusters_file']

    if 'Base' in handler_:
        # don't need vectors, this is a non-distributional experiment
        return result
    if vectors_path and clusters_path:
        raise ValueError('Cannot use both word vectors and word clusters')

    if random_thes and dummy_thes:
        raise ValueError('Cant use both random and dummy thesauri')
    elif random_thes:
        result['vector_source'] = RandomThesaurus(k=conf['vectorizer']['k'])
    elif dummy_thes:
        result['vector_source'] = DummyThesaurus()
    else:
        if vectors_path and clusters_path:
            raise ValueError('Cannot use both word vectors and word clusters')
        if 'signified' in handler_.lower() or vectors_exist:
            # vectors are needed either at decode time (signified handler) or during feature selection
            if not (vectors_path or clusters_path):
                raise ValueError('You must provide at least one source of distributional information '
                                 'because you requested %s and must_be_in_thesaurus=%s' % (handler_, vectors_exist))

    if len(vectors_path) == 1:
        # set up a row filter, if needed
        entries = vs_params['entries_of']
        if entries:
            entries = get_thesaurus_entries(entries)
            vs_params['row_filter'] = lambda x, y: x in entries
        if conf['vector_sources']['is_thesaurus']:
            result['vector_source'] = Thesaurus.from_tsv(vectors_path[0], **vs_params)
        else:
            result['vector_source'] = Vectors.from_tsv(vectors_path[0], **vs_params)
    if len(vectors_path) > 1:
        all_vect = [Vectors.from_tsv(p, **vs_params) for p in vectors_path]
        result['vector_source'] = MultiVectors(all_vect)

    if clusters_path:
        result['clusters'] = pd.read_hdf(clusters_path, key='clusters')

    return result


def get_thesaurus_entries(tsv_file):
    """
    Returns the set of entries contained in a thesaurus
    :param tsv_file: path to vectors file
    """
    return set(Vectors.from_tsv(tsv_file).keys())


def get_all_corpora():
    """
    Returns a manually compiled list of all corpora used in experiments. Each entry is a tuple of
    (short_name, path_on_disk)

    :rtype: list
    """
    return [('web', 'data/web-tagged'),
            ('amazon', 'data/amazon-xml'),
            ('maas', 'data/maas-xml')
            ]
