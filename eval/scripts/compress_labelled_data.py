import sys

sys.path.append('.')
import json
import gzip
import argparse
import logging
from joblib import Parallel, delayed
from eval.utils.data_utils import get_all_corpora, get_tokenizer_settings_from_conf_file, get_tokenized_data
from eval.plugins.bov import ThesaurusVectorizer
from eval.__main__ import is_valid_file


def jsonify_single_labelled_corpus(corpus_path, conf_file=None,
                                   tokenizer_conf=None,
                                   unigram_features=set('JNV'),
                                   phrase_features=set(['AN', 'NN', 'VO', 'SVO'])):
    """
    Tokenizes an entire XML corpus (sentence segmented and dependency parsed), incl test and train chunk,
    and writes its content to a single JSON gzip-ed file,
    one document per line. Each line is a JSON array, the first value of which is the label of
    the document, and the rest are JSON representation of a list of lists, containing all document
    features of interest, e.g. nouns, adj, NPs, VPs, wtc.
    The resultant document can be loaded with a GzippedJsonTokenizer.

    :param corpus_path: path to the corpus
    """

    def _write_corpus_to_json(x_tr, y_tr, outfile):
        vect = ThesaurusVectorizer(min_df=1,
                                   train_time_opts={'extract_unigram_features': unigram_features,
                                                    'extract_phrase_features': phrase_features})
        vect.extract_unigram_features = vect.train_time_opts['extract_unigram_features']
        vect.extract_phrase_features = vect.train_time_opts['extract_phrase_features']
        all_features = []
        for doc in x_tr:
            all_features.append([str(f) for f in vect.extract_features_from_token_list(doc)])

        for document, label in zip(all_features, y_tr):
            outfile.write(bytes(json.dumps([label, document]), 'UTF8'))
            outfile.write(bytes('\n', 'UTF8'))

    # load the dataset from XML/JSON/CoNLL
    if conf_file:
        conf = get_tokenizer_settings_from_conf_file(conf_file)
    elif tokenizer_conf:
        conf = tokenizer_conf
    else:
        raise ValueError('Must provide a dict or a file containing tokenizer config')
    x_tr, y_tr, x_test, y_test = get_tokenized_data(corpus_path, conf)

    with gzip.open('%s.gz' % corpus_path, 'wb') as outfile:
        _write_corpus_to_json(x_tr, y_tr, outfile)
        logging.info('Writing %s to gzip json', corpus_path)
        if x_test:
            _write_corpus_to_json(x_test, y_test, outfile)


def jsonify_all_labelled_corpora(n_jobs, conf):
    corpora = get_all_corpora()
    logging.info(corpora)
    Parallel(n_jobs=n_jobs)(delayed(jsonify_single_labelled_corpus)(corpus, conf) for corpus in corpora)


if __name__ == '__main__':
    """

    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s")

    parser = argparse.ArgumentParser()

    parser.add_argument('--conf', type=is_valid_file, required=True,
                        help='Conf file that contains the parameters of the tokenizer')

    parser.add_argument('--jobs', type=int, default=4,
                        help='Number of concurrent jobs')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', default=False,
                       help='Whether to compress ALL available labelled data sets or just one at a time')

    group.add_argument('--id', type=int,
                       help='If labelled data, compress just the labelled corpus at this position '
                            'in the predefined list. If unlabelled compress just '
                            'this thesaurus id in the database (must have been populated)')

    parameters = parser.parse_args()
    if parameters.all:
        jsonify_all_labelled_corpora(parameters.jobs, parameters.conf)
    else:
        jsonify_single_labelled_corpus(get_all_corpora()[parameters.id], parameters.conf)
