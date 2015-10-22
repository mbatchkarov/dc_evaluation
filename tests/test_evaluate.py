# smoke tests for various use cases of evaluate.py
import pytest
import pandas as pd

from discoutils.misc import mkdirs_if_not_exists
from eval.evaluate import run_experiment
from eval.utils.conf_file_utils import parse_config_file

base_handler = 'eval.pipeline.feature_handlers.BaseFeatureHandler'
hybrid_handler = 'eval.pipeline.feature_handlers.SignifierSignifiedFeatureHandler'
extreme_handler = 'eval.pipeline.feature_handlers.SignifiedOnlyFeatureHandler'

conf_file = 'eval/resources/conf/exp0/exp0.conf'
output_file = 'eval/resources/conf/exp0/output/tests-exp0.scores.csv'


@pytest.fixture
def conf():
    config, _ = parse_config_file(conf_file)
    mkdirs_if_not_exists(config['output_dir'])
    return config


def test_extreme_feature_expansion(conf):
    conf['feature_extraction']['decode_token_handler'] = extreme_handler
    run_experiment(conf)


def test_std_feature_expansion(conf):
    conf['feature_extraction']['decode_token_handler'] = hybrid_handler
    run_experiment(conf)


def test_nondistributional_baseline(conf):
    conf['feature_extraction']['decode_token_handler'] = base_handler
    conf['vector_sources']['neighbours_file'] = []
    conf['feature_selection']['must_be_in_thesaurus'] = False
    run_experiment(conf)


def test_nondistributional_baseline_improperly_configured(conf):
    conf['feature_extraction']['decode_token_handler'] = base_handler
    conf['vector_sources']['neighbours_file'] = []

    # we ask features to be in thesaurus, but do not provide one
    conf['feature_selection']['run'] = True
    conf['feature_selection']['must_be_in_thesaurus'] = True
    with pytest.raises(ValueError):
        run_experiment(conf)


def test_nondistributional_baseline_test_on_training_data(conf):
    conf['feature_extraction']['decode_token_handler'] = base_handler
    conf['vector_sources']['neighbours_file'] = []
    conf['crossvalidation']['type'] = 'oracle'
    conf['test_data'] = None

    run_experiment(conf)

    df = pd.read_csv(output_file, header=0)
    assert set(df.score) == {1.0}, 'Must achieve perfect accuracy'
