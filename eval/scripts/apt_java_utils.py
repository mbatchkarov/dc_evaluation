from glob import glob
import logging
import os

from discoutils.tokens import DocumentFeature
from joblib import Parallel
from joblib import delayed

from discoutils.collections_utils import walk_nonoverlapping_pairs
from discoutils.reduce_dimensionality import do_svd
from discoutils.thesaurus_loader import Vectors
from eval.scripts.compress_labelled_data import get_all_document_features

__author__ = 'mmb28'
USE_POS = False


def format_phrases():
    """
    Convert list of NPs in labelled corpora to CoNLL-like format for composition
    with the APT Java tool. Run this function before the Java code
    """
    feats = get_all_document_features(include_unigrams=False)
    logging.info('Loaded %d features', len(feats))
    pattern = '1\t{}\t2\t{}\n2\t{}\t0\troot\n\n'
    logging.info('Writing...')
    with open('features_in_labelled/apt-nps-to-compose.txt', 'w') as outfile:
        for df in feats:
            if df.type not in {'AN', 'NN'}:
                continue

            modifier, head = df.tokens
            # see http://universaldependencies.github.io/docs/u/dep/index.html
            relation = 'amod' if df.type == 'AN' else 'compound'
            if USE_POS:
                txt = pattern.format(modifier, relation, head)
            else:
                txt = pattern.format(modifier.text, relation, head.text)
            outfile.write(txt)


def _read_vector(f):
    bn = os.path.basename(f)
    sent_file = os.path.join(os.path.dirname(f),
                             '%s.sent' % bn.split('.')[0])
    with open(sent_file) as infile:
        phrase = ' '.join(line.strip().split('\t')[1] for line in infile if line.strip())

    with open(f) as infile:
        file_content = infile.readline().strip().split('\t')
    features = [(DocumentFeature.smart_lower(word, lowercasing=True), float(count))
                for (word, count) in walk_nonoverlapping_pairs(file_content, beg=0)]
    return phrase, features


def merge_vectors():
    d = {}
    files = glob('features_in_labelled/apt-nps-to-compose-small-composed/*vectorized')
    logging.info('Found %d composed phrase files', len(files))

    for phrase, features in Parallel(n_jobs=2)(delayed(_read_vector)(f) for f in files):
        d[phrase] = features

    logging.info('Successfully read %d APT vectors, converting to DiscoUtils data structure', len(d))
    v = Vectors(d)

    do_svd(v, 'apt-composed-vectors', reduce_to=[10], desired_counts_per_feature_type=None,
           write=1, use_hdf=False)

    # todo something like
    # do_svd(v, 'apt-composed-vectors', reduce_to=[10], desired_counts_per_feature_type=None,
    #        apply_to='path/to/unigram/vectors', write=3, use_hdf=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s",
                        datefmt='%m-%d %H:%M')
    # format_phrases()
    merge_vectors()
