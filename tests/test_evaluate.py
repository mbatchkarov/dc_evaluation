# smoke tests for various use cases of evaluate.py
import pytest
import pandas as pd
import numpy as np

from discoutils.misc import mkdirs_if_not_exists
from discoutils.thesaurus_loader import DenseVectors
from eval.evaluate import run_experiment
from eval.pipeline.feature_extractors import FeatureExtractor
from eval.scripts.kmeans_disco import cluster_vectors
from eval.utils.conf_file_utils import parse_config_file
from eval.utils.data_utils import get_tokenized_data

base_handler = 'eval.pipeline.feature_handlers.BaseFeatureHandler'
hybrid_handler = 'eval.pipeline.feature_handlers.SignifierSignifiedFeatureHandler'
extreme_handler = 'eval.pipeline.feature_handlers.SignifiedOnlyFeatureHandler'

conf_file = 'tests/resources/conf/exp0/exp0.conf'
output_file = 'tests/resources/conf/exp0/output/tests-exp0.scores.csv'


@pytest.fixture
def conf():
    config, _ = parse_config_file(conf_file)
    mkdirs_if_not_exists(config['output_dir'])
    return config


def test_extreme_feature_expansion(conf):
    conf['vectorizer']['decode_token_handler'] = extreme_handler
    run_experiment(conf)


def test_std_feature_expansion(conf):
    conf['vectorizer']['decode_token_handler'] = hybrid_handler
    run_experiment(conf)


def test_nondistributional_baseline(conf):
    conf['vectorizer']['decode_token_handler'] = base_handler
    conf['vector_sources']['neighbours_file'] = []
    conf['feature_selection']['must_be_in_thesaurus'] = False

    for debug_level in [0, 1, 2]:
        conf['debug_level'] = debug_level
        run_experiment(conf)


def test_nondistributional_baseline_improperly_configured(conf):
    conf['vectorizer']['decode_token_handler'] = base_handler
    conf['vector_sources']['neighbours_file'] = []

    # we ask features to be in thesaurus, but do not provide one
    conf['feature_selection']['run'] = True
    conf['feature_selection']['must_be_in_thesaurus'] = True
    with pytest.raises(ValueError):
        run_experiment(conf)


def test_nondistributional_baseline_test_on_training_data(conf):
    conf['vectorizer']['decode_token_handler'] = base_handler
    conf['vector_sources']['neighbours_file'] = []
    conf['crossvalidation']['type'] = 'oracle'
    conf['test_data'] = None

    run_experiment(conf)

    df = pd.read_csv(output_file, header=0)
    assert set(df.score) == {1.0}, 'Must achieve perfect accuracy'


def test_distributional_with_vector_clusters(conf, tmpdir):
    # generate random vectors for the the appropriate features and cluster them first
    x_tr, _, _, _ = get_tokenized_data(conf['training_data'], conf['tokenizer'])
    feats = FeatureExtractor().extract_features_from_tree_list([foo[0] for foo in x_tr])
    vectors = np.random.random((len(feats), 10))
    v = DenseVectors(pd.DataFrame(vectors, index=feats))
    tmpfile = str(tmpdir.join('tmp_random_vectors'))
    v.to_tsv(tmpfile, dense_hd5=True)

    tmpclusters = str(tmpdir.join('tmp_random_clusters'))
    cluster_vectors(tmpfile, tmpclusters, n_clusters=5, n_jobs=1)

    conf['vector_sources']['neighbours_file'] = []
    conf['vectorizer']['class'] = 'eval.pipeline.multivectors.KmeansVectorizer'
    conf['vector_sources']['clusters_file'] = tmpclusters
    # the features of the document are cluster ids, not phrases
    # no point in checking in they are in the thesaurus
    conf['feature_selection']['must_be_in_thesaurus'] = False

    for debug_level in [0, 1, 2]:
        conf['debug_level'] = debug_level
        run_experiment(conf)
