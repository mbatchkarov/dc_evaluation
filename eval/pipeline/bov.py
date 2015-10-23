# coding=utf-8
from collections import defaultdict
import logging
import array
import numbers

import networkx as nx
import scipy.sparse as sp
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from eval.pipeline.classifiers import NoopTransformer
from eval.pipeline.feature_extractors import FeatureExtractor
from eval.pipeline.feature_handlers import get_token_handler
from eval.pipeline.stats import get_stats_recorder


class ThesaurusVectorizer(TfidfVectorizer):
    """
    A thesaurus-backed CountVectorizer that can replace unknown features with
    their k nearest neighbours in the thesaurus.
    """

    def __init__(self, lowercase=True,
                 input='content', encoding='utf-8', decode_error='strict',
                 strip_accents=None,
                 preprocessor=None, analyzer='ngram',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 max_df=1.0, min_df=0,
                 max_features=None, vocabulary=None, binary=False, dtype=float,
                 norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False, use_tfidf=True,
                 debug_level=0, k=1,
                 sim_compressor='eval.utils.misc.unit',
                 train_token_handler='eval.pipeline.feature_handlers.BaseFeatureHandler',
                 decode_token_handler='eval.pipeline.feature_handlers.BaseFeatureHandler',
                 train_time_opts={'extract_unigram_features': ['J', 'N'],
                                  'extract_phrase_features': ['AN', 'NN', 'VO', 'SVO']},
                 decode_time_opts={'extract_unigram_features': '',
                                   'extract_phrase_features': ['AN', 'NN']},
                 standard_ngram_features=0,
                 remove_features_with_NER=False,
                 random_neighbour_thesaurus=False, **kwargs
                 ):
        """
        Builds a vectorizer the way a TfidfVectorizer is built, and takes one
        extra param specifying the path the the Byblo-generated thesaurus.

        Do not do any real work here, this constructor is first invoked with
        no parameters by the pipeline and then all the right params are set
        through reflection. When __init__ terminates the class invariants
        have not been established, make sure to check& establish them in
        fit_transform()

        :param standard_ngram_features: int. Extract standard (adjacent) ngram features up to this length
        """
        self.use_tfidf = use_tfidf
        self.debug_level = debug_level
        self.k = k
        self.sim_compressor = sim_compressor
        self.train_token_handler = train_token_handler
        self.decode_token_handler = decode_token_handler
        self.train_time_opts = train_time_opts
        self.decode_time_opts = decode_time_opts
        self.random_neighbour_thesaurus = random_neighbour_thesaurus

        self.stats = None
        self.handler = None
        self.feature_extractor = FeatureExtractor(remove_features_with_NER=remove_features_with_NER,
                                                  standard_ngram_features=standard_ngram_features,
                                                  **train_time_opts)

        super(ThesaurusVectorizer, self).__init__(input=input,
                                                  encoding=encoding,
                                                  decode_error=decode_error,
                                                  strip_accents=strip_accents,
                                                  lowercase=lowercase,
                                                  preprocessor=preprocessor,
                                                  analyzer=analyzer,
                                                  stop_words=stop_words,
                                                  token_pattern=token_pattern,
                                                  max_df=max_df,
                                                  min_df=min_df,
                                                  max_features=max_features,
                                                  vocabulary=vocabulary,
                                                  use_idf=use_idf,
                                                  smooth_idf=smooth_idf,
                                                  sublinear_tf=sublinear_tf,
                                                  binary=binary,
                                                  norm=norm,
                                                  dtype=dtype)

    def fit_transform(self, raw_documents, y=None, vector_source=None, stats_hdf_file=None, cv_fold=-1):
        self.cv_fold = cv_fold
        self.feature_extractor.update(**self.train_time_opts)
        self.thesaurus = vector_source
        self.handler = get_token_handler(self.train_token_handler,
                                         self.k,
                                         self.sim_compressor,
                                         self.thesaurus)
        # requested stats that to go HDF, store the name so we can record stats to that name at decode time too
        self.stats_hdf_file_ = stats_hdf_file
        self.stats = get_stats_recorder(self.debug_level, stats_hdf_file, 'tr', cv_fold, self.k)
        # a different stats recorder will be used for the testing data

        # ########## BEGIN super.fit_transform ##########
        # this is a modified version of super.fit_transform which works with an empty vocabulary
        self._validate_vocabulary()
        max_df = self.max_df
        min_df = self.min_df
        max_features = self.max_features

        vocabulary, X = self._count_vocab(raw_documents, self.fixed_vocabulary_)
        X = X.tocsc()

        if self.binary:
            X.data.fill(1)

        if not self.fixed_vocabulary_:
            if vocabulary:
                X = self._sort_features(X, vocabulary)

                n_doc = X.shape[0]
                max_doc_count = (max_df
                                 if isinstance(max_df, numbers.Integral)
                                 else int(round(max_df * n_doc)))
                min_doc_count = (min_df
                                 if isinstance(min_df, numbers.Integral)
                                 else int(round(min_df * n_doc)))
                if max_doc_count < min_doc_count:
                    raise ValueError(
                        "max_df corresponds to < documents than min_df")
                X, self.stop_words_ = self._limit_features(X, vocabulary,
                                                           max_doc_count,
                                                           min_doc_count,
                                                           max_features)

            self.vocabulary_ = vocabulary
        # ######### END super.fit_transform ##########
        if (self.thesaurus and hasattr(self.thesaurus, 'get_nearest_neighbours') and
                hasattr(self.thesaurus.get_nearest_neighbours, 'cache_info')):
            logging.info('NN cache info: %s', self.thesaurus.get_nearest_neighbours.cache_info())
        logging.info('Matrix shape is %r after vectorization', X.shape)
        return X, self.vocabulary_

    def transform(self, raw_documents):
        self.feature_extractor.update(**self.decode_time_opts)
        if not hasattr(self, 'vocabulary_'):
            self._check_vocabulary()

        if not hasattr(self, 'vocabulary_') or len(self.vocabulary_) == 0:
            raise ValueError("Vocabulary wasn't fitted or is empty!")
        # record stats separately for the test set
        self.stats = get_stats_recorder(self.debug_level, self.stats_hdf_file_, 'ev',
                                        self.cv_fold, self.k)

        if self.random_neighbour_thesaurus:
            # this is a bit of hack and a waste of effort, since a thesaurus will have been loaded first
            logging.info('Building random neighbour vector source with vocabulary of size %d', len(self.vocabulary_))
            self.thesaurus.k = self.k
            self.thesaurus.vocab = list(self.vocabulary_.keys())

        self.handler = get_token_handler(self.decode_token_handler,
                                         self.k,
                                         self.sim_compressor,
                                         self.thesaurus)

        # todo can't populate at this stage of the pipeline, because the vocabulary might
        # change if feature selection is enabled. Trying to do this will result in attempts to compose
        # features that we do not know how to compose because these have not been removed by FS
        # if self.thesaurus:
        # logging.info('Populating vector source %s prior to transform', self.thesaurus)
        # self.thesaurus.populate_vector_space(self.vocabulary_.keys())

        # BEGIN a modified version of super.transform that works when vocabulary is empty
        _, X = self._count_vocab(raw_documents, fixed_vocab=True)
        if self.binary:
            X.data.fill(1)
            # END super.transform

        if (self.thesaurus and hasattr(self.thesaurus, 'get_nearest_neighbours') and
                hasattr(self.thesaurus.get_nearest_neighbours, 'cache_info')):
            logging.info('NN cache info: %s', self.thesaurus.get_nearest_neighbours.cache_info())
        return X, self.vocabulary_

    def _extract_or_filter(self, thing):
        """
        Extract feature from a document. The document can be:
          - a list of sentences, each stored as a dependency parse tree, in which case features are extracted online.
            This can be a little slow.
          - a very broad list of pre-extracted features (list of str), which are just filtered. This is faster.
        """
        if isinstance(thing[0], str):
            # assume input already feature-extracted
            return self.feature_extractor.filter_preextracted_features(thing)
        if isinstance(thing[0], nx.DiGraph):
            # assume input tokenized, but features have not been extracted
            # corpora used in unit tests are like this
            return self.feature_extractor.extract_features_from_tree_list(thing)

    def _count_vocab(self, raw_documents, fixed_vocab):
        """
        Modified from sklearn 0.14's CountVectorizer

        @params fixed_vocab True if the vocabulary attribute has been set, i.e. the vectorizer is trained
        """
        if hasattr(self, 'cv_number'):
            logging.info('cv_number=%s', self.cv_number)
        logging.info('Converting features to vectors (with thesaurus lookup)')

        if not self.use_tfidf:
            self._tfidf = NoopTransformer()

        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # Add a new value when a new vocabulary item is seen, we're training now
            vocabulary = defaultdict(None)
            vocabulary.default_factory = vocabulary.__len__

        j_indices = array.array(str("i"))
        indptr = array.array(str("i"))
        values = []
        indptr.append(0)
        for doc_id, doc in enumerate(raw_documents):
            if doc_id % 1000 == 0:
                logging.info('Done %d/%d documents...', doc_id, len(raw_documents))
            for feature in self._extract_or_filter(doc):
                # ####################  begin non-original code  #####################
                self._process_single_feature(feature, j_indices, values, vocabulary)
            indptr.append(len(j_indices))

        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary)
            if not vocabulary:
                logging.error('Empty vocabulary')
        # some Python/Scipy versions won't accept an array.array:
        if j_indices:
            j_indices = np.frombuffer(j_indices, dtype=np.intc)
        else:
            j_indices = np.array([], dtype=np.int32)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        # values = np.ones(len(j_indices))

        X = sp.csr_matrix((values, j_indices, indptr),
                          shape=(len(indptr) - 1, len(vocabulary)),
                          dtype=self.dtype)
        X.sum_duplicates()  # nice that the summation is explicit
        self.stats.consolidate_stats()
        return vocabulary, X

    def _process_single_feature(self, feature, j_indices, values, vocabulary):
        try:
            feature_index_in_vocab = vocabulary[feature]
        except KeyError:
            feature_index_in_vocab = None
            # if term is not in seen vocabulary
        # is_in_vocabulary = bool(feature_index_in_vocab is not None)
        is_in_vocabulary = feature in vocabulary
        # is_in_th = bool(self.thesaurus.get(feature))
        is_in_th = feature in self.thesaurus if self.thesaurus else False
        self.stats.register_token(feature, is_in_vocabulary, is_in_th)
        # j_indices.append(feature_index_in_vocab) # todo this is the original code, also updates vocabulary
        params = {'feature': feature,
                  'feature_index_in_vocab': feature_index_in_vocab,
                  'vocabulary': vocabulary, 'j_indices': j_indices,
                  'values': values, 'stats': self.stats}
        if is_in_vocabulary and is_in_th:
            self.handler.handle_IV_IT_feature(**params)
        if is_in_vocabulary and not is_in_th:
            self.handler.handle_IV_OOT_feature(**params)
        if not is_in_vocabulary and is_in_th:
            self.handler.handle_OOV_IT_feature(**params)
        if not is_in_vocabulary and not is_in_th:
            self.handler.handle_OOV_OOT_feature(**params)
            #####################  end non-original code  #####################
