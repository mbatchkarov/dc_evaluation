from collections import Counter
import sys
import os

from discoutils.tokens import DocumentFeature

from discoutils.misc import mkdirs_if_not_exists
from eval.pipeline.feature_extractors import FeatureExtractor

sys.path.append('.')
import json
import gzip
import argparse
import logging
from joblib import Parallel, delayed
from eval.utils.data_utils import get_all_corpora, get_tokenizer_settings_from_conf_file, get_tokenized_data
from eval.evaluate import is_valid_file

ROOT = 'features_in_labelled'


def get_all_document_features(include_unigrams=False, remove_pos=False):
    """
    Finds all noun-noun and adj-noun compounds (and optionally adjs and nouns) in all labelled corpora
    mentioned in the conf files.
    :param include_unigrams: if False, only NPs will be returned
    :param remove_pos: whether to remove PoS tags if present, result will be either "cat/N" or "cat"
    :rtype: set of DocumentFeature
    """
    result = set()
    accepted_df_types = {'AN', 'NN', 'VO', 'SVO', '1-GRAM'} if include_unigrams else {'AN', 'NN', 'VO', 'SVO'}
    for corpus_name, _ in get_all_corpora():
        path = os.path.abspath(os.path.join(__file__, '..', '..', '..', ROOT, '%s_all_features.txt' % corpus_name))
        with open(path) as infile:
            for line in infile:
                df = DocumentFeature.from_string(line.strip())
                if df.type in accepted_df_types:
                    if remove_pos:
                        # todo these are of type str, in the other branch it's DocumentFeature. things will likely break
                        result.add(df.ngram_separator.join(t.text for t in df.tokens))
                    else:
                        result.add(df)

    logging.info('Found a total of %d features in all corpora', len(result))
    if not remove_pos:
        logging.info('Their types are %r', Counter(df.type for df in result))
    if include_unigrams:
        logging.info('PoS tags of unigrams are are %r',
                     Counter(df.tokens[0].pos for df in result if df.type == '1-GRAM'))
    else:
        logging.info('Unigram features not included!')
    return result


def _write_features_of_single_corpus_to_file(all_phrases, corpus_name):
    ALL_FEATURES_FILE = '%s/%s_all_features.txt' % (ROOT, corpus_name)
    NP_MODIFIERS_FILE = '%s/%s_np_modifiers.txt' % (ROOT, corpus_name)
    VERBS_FILE = '%s/%s_verbs.txt' % (ROOT, corpus_name)
    SOCHER_FILE = '%s/%s_socher.txt' % (ROOT, corpus_name)

    logging.info('Writing %d unique document features to files in %s', len(all_phrases), ROOT)

    # How stanford parser formats NPs and VPs
    # (ROOT
    # (NP (NN acquisition) (NN pact)))
    #
    # (ROOT
    # (NP (JJ pacific) (NN stock)))
    stanford_NP_pattern = '(ROOT\n (NP ({} {}) ({} {})))\n\n'

    # (ROOT
    # (S
    # (NP (NNS cats))
    # (VP (VBP eat)
    # (NP (NNS dogs)))))
    stanford_SVO_pattern = '(ROOT\n  (S\n    (NP (NN {}))\n    (VP (VB {})\n      (NP (NN {})))))\n\n'

    # (ROOT
    # (S
    # (VP (VB eat)
    # (NP (NNS cats)))))
    stanford_VO_pattern = '(ROOT\n  (S\n    (VP (VB {})\n      (NP (NN {})))))\n\n'

    # (ROOT
    # (NP (NN roads)))
    # I checked that this extracts the neural word embedding for the word
    stanford_unigram_pattern = '(ROOT\n (NP ({} {})))\n\n'

    mkdirs_if_not_exists(ROOT)
    logging.info('Writing all document features to files')
    seen_modifiers, seen_verbs = set(), set()

    with open(SOCHER_FILE, 'w') as outf_socher, \
            open(NP_MODIFIERS_FILE, 'w') as outf_mods, \
            open(VERBS_FILE, 'w') as outf_verbs, \
            open(ALL_FEATURES_FILE, 'w') as outf_plain:

        for item in all_phrases:
            item = DocumentFeature.from_string(item)
            # write in my underscore-separated format
            outf_plain.write(str(item) + '\n')

            if item.type in {'AN', 'NN'}:
                # write the phrase in Socher's format
                string = stanford_NP_pattern.format(item.tokens[0].pos * 2, item.tokens[0].text,
                                                    item.tokens[1].pos * 2, item.tokens[1].text)
                outf_socher.write(string)

            if item.type in {'VO', 'SVO'}:
                verb = str(item.tokens[-2])
                if verb not in seen_verbs:
                    seen_verbs.add(verb)
                    outf_verbs.write(verb)
                    outf_verbs.write('\n')

            if item.type == 'VO':
                string = stanford_VO_pattern.format(*[x.tokens[0].text for x in item])
                outf_socher.write(string)

            if item.type == 'SVO':
                string = stanford_SVO_pattern.format(*[x.tokens[0].text for x in item])
                outf_socher.write(string)

            if item.type in {'AN', 'NN'}:
                # write just the modifier separately
                first = str(item.tokens[0])
                second = str(item.tokens[1])
                if first not in seen_modifiers:
                    outf_mods.write('%s\n' % first)
                    seen_modifiers.add(first)

            if item.type == '1-GRAM':
                string = stanford_unigram_pattern.format(item.tokens[0].pos * 2, item.tokens[0].text)
                outf_socher.write(string)

            if item.type not in {'1-GRAM', 'AN', 'NN', 'VO', 'SVO'}:  # there shouldn't be any other features
                raise ValueError('Item %r has the wrong feature type: %s' % (item, item.type))


def jsonify_single_labelled_corpus(corpus_name, corpus_path,
                                   conf_file=None,
                                   tokenizer_conf=None,
                                   unigram_features=set('JNV'),
                                   phrase_features=set(['AN', 'NN', 'VO', 'SVO']),
                                   write_feature_set=False):
    """
    Tokenizes an entire XML/CoNLL corpus (sentence segmented and dependency parsed), incl test and train chunk,
    and writes its content to a single JSON gzip-ed file,
    one document per line. Each line is a JSON array, the first value of which is the label of
    the document, and the rest are JSON representation of a list of lists, containing all document
    features of interest, e.g. nouns, adj, NPs, VPs, wtc.
    The resultant document can be loaded with a GzippedJsonTokenizer.

    :param corpus_path: path to the corpus
    """

    def _write_corpus_to_json(x_tr, y_tr):
        extr = FeatureExtractor(extract_unigram_features=unigram_features,
                                extract_phrase_features=phrase_features)
        documents = []
        for doc in x_tr:
            documents.append([str(f) for f in extr.extract_features_from_tree_list(doc)])

        for document, label in zip(documents, y_tr):
            outfile.write(bytes(json.dumps([label, document]), 'UTF8'))
            outfile.write(bytes('\n', 'UTF8'))

        return set(feat for doc in documents for feat in doc)

    # load the dataset from XML/JSON/CoNLL
    if conf_file:
        conf = get_tokenizer_settings_from_conf_file(conf_file)
    elif tokenizer_conf:
        conf = tokenizer_conf
    else:
        raise ValueError('Must provide a dict or a file containing tokenizer config')
    x_tr, y_tr, x_test, y_test = get_tokenized_data(corpus_path, conf)

    with gzip.open('%s.gz' % corpus_path, 'wb') as outfile:
        feats = _write_corpus_to_json(x_tr, y_tr)
        logging.info('Writing %s to gzip json', corpus_path)
        if x_test:
            feats |= _write_corpus_to_json(x_test, y_test)

    if write_feature_set:
        _write_features_of_single_corpus_to_file(feats, corpus_name)


def jsonify_all_labelled_corpora(n_jobs, *args, **kwargs):
    corpora = get_all_corpora()
    logging.info('Converting the following corpora to JSON: %r', [c[0] for c in corpora])
    Parallel(n_jobs=n_jobs)(delayed(jsonify_single_labelled_corpus)(*(path + args), **kwargs) for path in corpora)


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

    parser.add_argument('--write-features', action='store_true', default=False,
                        help='Whether to store a set of all features in a range of formats')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', default=False,
                       help='Whether to compress ALL available labelled data sets or just one at a time')

    group.add_argument('--id', type=int,
                       help='If labelled data, compress just the labelled corpus at this position '
                            'in the predefined list. If unlabelled compress just '
                            'this thesaurus id in the database (must have been populated)')

    parameters = parser.parse_args()
    if parameters.all:
        jsonify_all_labelled_corpora(parameters.jobs, parameters.conf,
                                     write_feature_set=parameters.write_features)
    else:
        jsonify_single_labelled_corpus(get_all_corpora()[parameters.id][1],
                                       parameters.conf,
                                       write_feature_set=parameters.write_features)
