import argparse
from collections import Counter
from glob import glob
import gzip
from itertools import zip_longest
import logging
import os
import time

from joblib import Parallel

from joblib import delayed

from sklearn.decomposition import TruncatedSVD

from discoutils.tokens import DocumentFeature

from discoutils.collections_utils import walk_nonoverlapping_pairs
from discoutils.io_utils import write_vectors_to_hdf
from discoutils.thesaurus_loader import Vectors

__author__ = 'mmb28'
USE_POS = False
ROOT = 'features_in_labelled'


# copied from dc-evaluation
def get_all_corpora():
    return [('web', 'data/web-tagged'),
            ('amazon', 'data/amazon-xml')
            ]


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


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


def _read_vector(vector_file):
    bn = os.path.basename(vector_file)
    sent_file = os.path.join(os.path.dirname(vector_file),
                             '%s.sent' % bn.split('.')[0])
    if not os.path.exists(sent_file):
        return '__MISSING__', {}

    with open(sent_file) as infile:
        phrase = ' '.join(line.strip().split('\t')[1] for line in infile if line.strip())

    with gzip.open(vector_file) as infile:
        file_content = infile.readline().decode('utf8').strip().split('\t')
    features = [(DocumentFeature.smart_lower(word, lowercasing=True), float(count))
                for (word, count) in walk_nonoverlapping_pairs(file_content, beg=0)]
    return phrase, features


def merge_vectors(composed_dir, unigrams, output, workers=4, chunk_size=10000):
    # this particular dataset uses spaces instead of underscores. State this to avoid parsing issues
    DocumentFeature.ngram_separator = ' '
    DIMS = 100  # SVD dimensionality

    d = {}
    files = glob(os.path.join(composed_dir, '*apt.vec.gz'))
    logging.info('Found %d composed phrase files', len(files))

    # ignore stuff that isn't unigrams, it will cause problems later
    unigrams = Vectors.from_tsv(unigrams, row_filter=lambda x, y: y.type == '1-GRAM')
    logging.info('Found %d unigram vectors', len(unigrams))

    mat, cols, rows = unigrams.to_sparse_matrix()
    cols = set(cols)
    svd = TruncatedSVD(DIMS, random_state=0)
    logging.info('Reducing dimensionality of matrix of shape %r...', mat.shape)
    start = time.time()
    reduced_mat = svd.fit_transform(mat)
    logging.info('Reduced using {} from shape {} to shape {} in {} seconds'.format(svd,
                                                                                   mat.shape,
                                                                                   reduced_mat.shape,
                                                                                   time.time() - start))
    write_vectors_to_hdf(reduced_mat, rows,
                         ['SVD:feat{0:03d}'.format(i) for i in range(reduced_mat.shape[1])],
                         '%s-unigrams-SVD%d' % (output, DIMS))
    del mat

    for i, chunk in enumerate(grouper(chunk_size, files)):
        logging.info('Reading composed vectors, chunk %d...', i)
        for phrase, features in Parallel(n_jobs=workers)(delayed(_read_vector)(f) for f in chunk if f):
            if features:
                d[phrase] = features

        logging.info('Found %d non-empty composed vectors in this chunk, running SVD now...', len(d))
        if not d:
            continue

        composed_vec = Vectors(d, column_filter=lambda foo: foo in cols)
        # vectorize second matrix with the vocabulary (columns) of the first thesaurus to ensure shapes match
        # "project" composed matrix into space of unigram thesaurus
        unigrams.v.vocabulary_ = {x: i for i, x in enumerate(list(cols))}
        extra_matrix = unigrams.v.transform([dict(fv) for fv in composed_vec.values()])
        assert extra_matrix.shape == (len(composed_vec), len(cols))
        logging.info('Composed matrix is of shape %r before SVD', extra_matrix.shape)

        extra_matrix = svd.transform(extra_matrix)
        write_vectors_to_hdf(extra_matrix,
                             list(composed_vec.keys()),
                             ['SVD:feat{0:03d}'.format(i) for i in range(extra_matrix.shape[1])],
                             '%s-phrases-chunk%d-SVD%d' % (output, i, DIMS))
        del composed_vec


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s")
    # format_phrases()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--unigrams', required=True, help='Unigram vectors file')
    parser.add_argument('--composed', required=True, help='Composed vector directory, as output by APT')
    parser.add_argument('--output', required=True, help='Name of output file. ')
    parser.add_argument('--workers', default=4, type=int, help='Worker process count, default=4')
    parser.add_argument('--chunk-size', default=1000, type=int, help='Number of composed vectors to read at a time')

    args = parser.parse_args()
    logging.info('Parameters are %r', args)
    merge_vectors(args.composed, args.unigrams, args.output, workers=args.workers, chunk_size=args.chunk_size)
