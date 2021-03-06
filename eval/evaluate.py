# !/usr/bin/python
# coding=utf-8
# if one tries to run this script from the main project directory the
# eval package would not be on the path, add it and try again
import sys

sys.path.append('.')
import argparse

import pickle
import os
import shutil
import logging
from datetime import datetime
from pandas import DataFrame
import numpy as np
from numpy.ma import hstack
from sklearn import cross_validation
from sklearn.pipeline import Pipeline

from discoutils.misc import Bunch, mkdirs_if_not_exists
from eval.utils.data_utils import get_pipeline_fit_args, get_tokenized_data, get_tokenizer_settings_from_conf
from eval.pipeline.classifiers import (LeaveNothingOut, PredefinedIndicesIterator,
                              SubsamplingPredefinedIndicesIterator, PicklingPipeline)
from eval.pipeline.feature_selectors import MetadataStripper
from eval.pipeline.dumpers import FeatureVectorsCsvDumper
from eval.utils.conf_file_utils import parse_config_file
from eval.utils.misc import multiple_scores, update_dict_according_to_mask
from eval.utils.reflection_utils import get_named_object
from eval.utils.reflection_utils import get_intersection_of_parameters


def _build_crossvalidation_iterator(config, y_train, y_test=None):
    """
    Returns a crossvalidation iterator, which contains a list of
    (train_indices, test_indices) that can be used to slice
    a dataset to perform crossvalidation. Additionally,
    returns the original data that was passed in and a mask specifying what
    data points should be used for validation.

    The method of splitting for CV is determined by what is specified in the
    conf file. The splitting of data in train/test/validate set is not done
    in this function- here we only return a mask for the validation data
    and an iterator for the train/test data.
    The full text is provided as a parameter so that joblib can cache the
    call to this function.
    """
    cv_type = config['type']
    k = config['k']
    dataset_size = len(y_train)

    if y_test is not None:
        logging.warning('You have requested test set to be used for evaluation.')
        if cv_type != 'test_set' and cv_type != 'subsampled_test_set':
            raise ValueError('Wrong crossvalidation type. Only test_set '
                             'or subsampled_test_set are permitted with a test set')

        train_indices = range(dataset_size)
        test_indices = range(dataset_size, dataset_size + len(y_test))
        y_train = hstack([y_train, y_test])
        dataset_size += len(y_test)

    random_state = config['random_state']
    if k < 0:
        logging.warning('crossvalidation.k not specified, defaulting to 1')
        k = 1
    if cv_type == 'kfold':
        iterator = cross_validation.KFold(dataset_size, n_folds=int(k), random_state=random_state)
    elif cv_type == 'skfold':
        iterator = cross_validation.StratifiedKFold(y_train, n_folds=int(k), random_state=random_state)
    elif cv_type == 'oracle':
        iterator = LeaveNothingOut(dataset_size)
    elif cv_type == 'test_set' and y_test is not None:
        iterator = PredefinedIndicesIterator(train_indices, test_indices)
    elif cv_type == 'subsampled_test_set' and y_test is not None:
        iterator = SubsamplingPredefinedIndicesIterator(y_train,
                                                        train_indices,
                                                        test_indices, int(k),
                                                        config['sample_size'],
                                                        config['random_state'])
    else:
        raise ValueError('Unrecognised crossvalidation type %(cv_type)s. The supported types are kfold, skfold, '
                         'test_set, subsampled_test_set and oracle')

    return iterator, y_train


def _build_vectorizer(init_args, conf, pipeline_list):
    """
    Builds a vectorized that converts raw text to feature vectors. The
    parameters for the vectorizer are specified in the *feature extraction*
    section of the configuration file. These will be matched to those of the
     *__init__* method of the    vectorizer class and the matching keywords
     are passed to the vectorizer. The non-matching arguments are simply
     ignored.

     The vectorizer converts a corpus into a term frequency matrix. A given
     source corpus is converted into a term frequency matrix and
     returned as a numpy *coo_matrix*.The value of the *vectorizer* field
     in the main configuration file is used as the transformer class. This class
     has to implement the methods *fit*, *transform* and *fit_transform* as
     per scikit-learn.
    """
    vectorizer = get_named_object(conf['vectorizer']['class'])

    # get the names of the arguments that the vectorizer class takes
    # the object must only take keyword arguments
    # todo KmeansVectorizer does not declare its parameters explicitly so intersection doesnt work
    # instead its constructor should take **kwargs, and we can pass in whatever we want with no need to manually check
    # which parameters are valid for that object
    init_args.update(get_intersection_of_parameters(vectorizer, conf['feature_extraction'], 'vect'))
    init_args.update(get_intersection_of_parameters(vectorizer, conf['vectorizer'], 'vect'))
    init_args.update(get_intersection_of_parameters(vectorizer, conf, 'vect')) # get debug_level from conf file
    pipeline_list.append(('vect', vectorizer()))


def _build_feature_selector(init_args, feature_selection_conf, pipeline_list):
    """
    If feature selection is required, this function appends a selector
    object to pipeline_list and its configuration to configuration. Note this
     function modifies (appends to) its input arguments
    """
    if feature_selection_conf['run']:
        method = get_named_object(feature_selection_conf['method'])
        scoring = feature_selection_conf.get('scoring_function')
        logging.info('Scoring function is %s', scoring)
        scoring_func = get_named_object(scoring) if scoring else None

        # the parameters for steps in the Pipeline are defined as
        # <component_name>__<arg_name> - the Pipeline (which is actually a
        # BaseEstimator) takes care of passing the correct arguments down
        # along the pipeline, provided there are no name clashes between the
        # keyword arguments of two consecutive transformers.

        init_args.update(get_intersection_of_parameters(method, feature_selection_conf, 'fs'))
        logging.info('FS method is %s', method)
        pipeline_list.append(('fs', method(scoring_func)))


def _build_pipeline(conf, cv_i):
    """
    Builds a pipeline consisting of
        - feature extractor
        - optional feature selection
        - optional dimensionality reduction
        - classifier
    """
    exp_name = conf['name']
    debug_level = conf['debug_level']

    init_args = {}
    pipeline_list = []

    _build_vectorizer(init_args, conf, pipeline_list)

    _build_feature_selector(init_args, conf['feature_selection'], pipeline_list)

    # put the optional dumper after feature selection/dim. reduction
    if debug_level > 0:
        logging.info('Will perform post-vectorizer data dump')
        pipeline_list.append(('dumper', FeatureVectorsCsvDumper(exp_name, cv_i, conf['output_dir'])))

    # vectorizer will return a matrix (as usual) and some metadata for use with feature dumper/selector,
    # strip them before we proceed to the classifier
    pipeline_list.append(('stripper', MetadataStripper()))

    fit_args = {}
    if debug_level > 0:
        fit_args['vect__stats_hdf_file'] = 'statistics/stats-%s' % exp_name

    pipeline = PicklingPipeline(pipeline_list, exp_name) if debug_level > 1 else Pipeline(pipeline_list)
    shared_fit_args = get_pipeline_fit_args(conf)
    for step_name, _ in pipeline.steps:
        for param_name, param_val in shared_fit_args.items():
            fit_args['%s__%s' % (step_name, param_name)] = param_val
        fit_args['%s__cv_fold' % step_name] = cv_i
    fit_args['stripper__strategy'] = conf['vector_sources']['neighbour_strategy']
    # tell vector source to retrieve a few more neighbours than would be needed
    fit_args['stripper__k'] = int(conf['vectorizer']['k'] * 8)
    pipeline.set_params(**init_args)
    logging.debug('Pipeline is:\n %s', pipeline)
    return pipeline, fit_args


def _build_classifiers(classifiers_conf):
    for i, clf_name in enumerate(classifiers_conf):
        if not classifiers_conf[clf_name]:
            continue
            # ignore disabled classifiers
        if not classifiers_conf[clf_name]['run']:
            logging.debug('Ignoring classifier %s' % clf_name)
            continue
        clf = get_named_object(clf_name)
        init_args = get_intersection_of_parameters(clf, classifiers_conf[clf_name])
        yield clf(**init_args)


def _cv_loop(config, cv_i, score_func, test_idx, train_idx, X, y):
    logging.info('Starting CV fold %d', cv_i)
    pipeline, fit_params = _build_pipeline(config, cv_i)

    # code below is a simplified version of sklearn's _cross_val_score
    train_text = [X[idx] for idx in train_idx]
    test_text = [X[idx] for idx in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # vectorize all data in advance, it's the same across all classifiers
    tr_matrix = pipeline.fit_transform(train_text, y_train, **fit_params)
    # Update the vocabulary of the vectorizer. The feature selector may remove some vocabulary entries,
    # but the vectorizer will be unaware of this. Because the vectorizer's logic is conditional on whether
    # features are IV or OOV, this is a problem. (The issue is caused by the fact that the vectorizer and feature
    # selector are not independent. The special logic of the vectorizer should have been implemented as a third
    # transformer, but this would require too much work at this time.
    if 'fs' in pipeline.named_steps:
        pipeline.named_steps['vect'].vocabulary_ = pipeline.named_steps['fs'].vocabulary_
    test_matrix = pipeline.transform(test_text)
    stats = pipeline.named_steps['vect'].stats

    # remove documents with too few features
    to_keep_train = tr_matrix.sum(axis=1) >= config['min_train_features']
    to_keep_train = np.ravel(np.array(to_keep_train))
    logging.info('%d/%d train documents have enough features', sum(to_keep_train), len(y_train))
    tr_matrix = tr_matrix[to_keep_train, :]
    y_train = y_train[to_keep_train]

    # the slice above may remove all occurrences of a feature,
    # e.g. when it only occurs in one document (very common) and the document
    # doesn't have enough features. Drop empty columns in the term-doc matrix
    column_mask = tr_matrix.sum(axis=0) > 0
    column_mask = np.squeeze(np.array(column_mask))
    tr_matrix = tr_matrix[:, column_mask]

    voc = update_dict_according_to_mask(pipeline.named_steps['vect'].vocabulary_, column_mask)
    inv_voc = {index: feature for (feature, index) in voc.items()}

    # do the same for the test set
    to_keep_test = test_matrix.sum(axis=1) >= config['min_test_features']  # todo need unit test
    to_keep_test = np.ravel(np.array(to_keep_test))
    logging.info('%d/%d test documents have enough features', np.count_nonzero(to_keep_test), len(y_test))
    test_matrix = test_matrix[to_keep_test, :]
    y_test = y_test[to_keep_test]

    np.savetxt(os.path.join(config['output_dir'], 'gold-cv%d.csv' % cv_i),
               y_test, delimiter=',', fmt="%s")

    scores_this_cv_run = []
    for clf in _build_classifiers(config['classifiers']):
        if not (np.count_nonzero(to_keep_train) and np.count_nonzero(to_keep_test)):
            logging.error('There isnt enough test data for a proper evaluation, skipping this fold!!!')
            continue  # if there's no training data or test data just ignore the fold
        logging.info('Starting training of %s', clf)
        clf = clf.fit(tr_matrix, y_train)
        predictions = clf.predict(test_matrix)
        scores = score_func(y_test, predictions)

        tr_set_scores = score_func(y_train, clf.predict(tr_matrix))
        logging.info('Training set scores: %r', tr_set_scores)
        clf_name = clf.__class__.__name__.split('.')[-1]
        np.savetxt(os.path.join(config['output_dir'], 'predictions-%s-cv%d.csv' % (clf_name, cv_i)),
                   predictions, delimiter=',', fmt="%s")

        if config['debug_level'] > 1:
            # if a feature selectors exist, use its vocabulary
            # step_name = 'fs' if 'fs' in pipeline.named_steps else 'vect'
            with open('%s.%s.pkl' % (stats.prefix, clf_name), 'wb') as outf:
                logging.info('Pickling trained classifier to %s', outf.name)
                b = Bunch(clf=clf, inv_voc=inv_voc, tr_matrix=tr_matrix,
                          test_matrix=test_matrix, predictions=predictions,
                          y_tr=y_train, y_ev=y_test, train_mask=to_keep_train,
                          test_mask=to_keep_test)
                pickle.dump(b, outf)

        for metric, score in scores.items():
            scores_this_cv_run.append(
                [type(clf).__name__,
                 cv_i,
                 metric.split('.')[-1],
                 score])
        logging.info('Done with %s', clf)
    logging.info('Finished CV fold %d', cv_i)
    try:
        v = shared_fit_args['vector_source']
        logging.info('Cache info: %s', v.get_nearest_neighbours.cache_info())
    except Exception:
        # can fail for a number of reasons, don't care much if it does
        pass
    return scores_this_cv_run


def _store_scores(scores, output_dir, name):
    """
    Stores a csv representation of the results
    """

    logging.info("Saving to %s", output_dir)
    if not scores:
        raise ValueError('Scores are missing for this experiment. This may be because feature selection removed all '
                         'test documents for all folds, and accuracy could not be computed')
    df = DataFrame(scores,
                   columns=['classifier', 'cv_no', 'metric', 'score'])
    csv = os.path.join(output_dir, '%s.scores.csv' % name)
    df.to_csv(csv, na_rep='-1')


def run_experiment(conf):
    start_time = datetime.now()
    mkdirs_if_not_exists(conf['output_dir'])
    test_path = ''
    tr_data = conf['training_data']
    if conf['test_data']:
        test_path = conf['test_data']

    # LOADING RAW TEXT
    x_tr, y_tr, x_test, y_test = get_tokenized_data(tr_data,
                                                    get_tokenizer_settings_from_conf(conf),
                                                    test_data=test_path)

    # CREATE CROSSVALIDATION ITERATOR
    cv_iterator, y_vals = _build_crossvalidation_iterator(conf['crossvalidation'],
                                                          y_tr, y_test)
    if x_test is not None:
        # concatenate all data, the CV iterator will make sure x_test is used for testing
        x_vals = list(x_tr)
        x_vals.extend(list(x_test))
    else:
        x_vals = x_tr

    all_scores = []
    params = []
    for i, (train_idx, test_idx) in enumerate(cv_iterator):
        params.append((conf, i, multiple_scores, test_idx, train_idx, x_vals, y_vals))
        logging.warning('Only using the first CV fold')
        if conf['crossvalidation']['break_after_first']:
            # only do one train/test split to save time
            logging.info('Exiting after first fold')
            break

    scores_over_cv = [_cv_loop(*foo) for foo in params]
    all_scores.extend([score for one_set_of_scores in scores_over_cv for score in one_set_of_scores])
    _store_scores(all_scores, conf['output_dir'], conf['name'])
    total_time = (datetime.now() - start_time).seconds / 60
    logging.info('MINUTES TAKEN %.2f' % total_time)


def is_valid_file(arg):
    if not os.path.exists(arg):
        parser.error("The conf file %s does not exist!" % arg)
    else:
        return arg


if __name__ == '__main__':
    # parse command-line arguments (conf file only)
    parser = argparse.ArgumentParser(description='Evaluate vector via document classification')
    parser.add_argument('conf_file',
                        help='Conf file that defines the experiment',
                        type=is_valid_file)

    args = parser.parse_args()
    conf, configspec_file = parse_config_file(args.conf_file)
    mkdirs_if_not_exists(conf['output_dir'])

    # set up logging to file
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s",
                        datefmt='%m-%d %H:%M',
                        filename=os.path.join(conf['output_dir'], 'log.txt'),
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter("%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s")
    # tell the handler to use this format
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info('Read configuration file from %s, conf spec from %s',
                 args.conf_file, configspec_file)
    shutil.copy(args.conf_file, conf['output_dir'])

    # run data through the pipeline
    run_experiment(conf)
    # consolidate_single_experiment(i) # todo show examples of naklar integration
