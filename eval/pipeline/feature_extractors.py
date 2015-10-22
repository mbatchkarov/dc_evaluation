import logging
from operator import attrgetter

from discoutils.tokens import DocumentFeature


class FeatureExtractor(object):
    def __init__(self, remove_features_with_NER=False,
                 extract_unigram_features='J,N',
                 extract_phrase_features=['AN', 'NN', 'SVO'],
                 standard_ngram_features=0):
        self.remove_features_with_NER = remove_features_with_NER
        self.extract_phrase_features = extract_phrase_features
        self.extract_unigram_features = extract_unigram_features
        self.standard_ngram_features = standard_ngram_features

        self.entity_ner_tags = {'ORGANIZATION', 'PERSON', 'LOCATION'}

    def update(self, **kwargs):
        self.__dict__.update(**kwargs)

    def extract_features_from_dependency_tree(self, parse_tree):
        new_features = []

        # extract sentence-internal adjective-noun compounds
        if 'AN' in self.extract_phrase_features:
            # get tuples of (head, dependent) for each amod relation in the tree
            # also enforce that head is a noun, dependent is an adjective
            for head, dep, data in parse_tree.edges(data=True):
                if data['type'] == 'amod' and head.pos == 'N' and dep.pos == 'J':
                    new_features.append(DocumentFeature('AN', (dep, head)))

        if 'VO' in self.extract_phrase_features or 'SVO' in self.extract_phrase_features:
            # extract sentence-internal subject-verb-direct object compounds
            # todo how do we handle prepositional objects?
            verbs = [t for t in parse_tree.nodes() if t.pos == 'V']

            objects = set([(head, dep) for head, dep, data in parse_tree.edges(data=True)
                           if data['type'] == 'dobj' and head.pos == 'V' and dep.pos == 'N'])

        if 'SVO' in self.extract_phrase_features:
            subjects = set([(head, dep) for head, dep, opts in parse_tree.edges(data=True) if
                            opts['type'] == 'nsubj' and head.pos == 'V' and dep.pos == 'N'])

            subjverbobj = [(s[1], v, o[1]) for v in verbs for s in subjects for o in objects if s[0] == v and o[0] == v]

            for s, v, o in subjverbobj:
                new_features.append(DocumentFeature('SVO', (s, v, o)))

        if 'SVO' in self.extract_phrase_features:
            verbobj = [(v, o[1]) for v in verbs for o in objects if o[0] == v]
            for v, o in verbobj:
                new_features.append(DocumentFeature('VO', (v, o)))

        if 'NN' in self.extract_phrase_features:
            for head, dep, data in parse_tree.edges(data=True):
                if data['type'] == 'nn' and head.pos == 'N' and dep.pos == 'N':
                    new_features.append(DocumentFeature('NN', (dep, head)))

        if self.remove_features_with_NER:
            return self._remove_features_containing_named_entities(new_features)
        return new_features

    def extract_features_from_token_list(self, doc_sentences):
        """
        Turn a document( a list of tokens) into a sequence of features.
        """
        features = []

        # extract sentence-internal token n-grams

        for parse_tree in doc_sentences:
            if not parse_tree:  # the sentence segmenter sometimes returns empty sentences
                continue

            if parse_tree:
                features.extend(self.extract_features_from_dependency_tree(parse_tree))
            else:
                # sometimes an input document will have a sentence of one word, which has no dependencies
                # just ignore that and extract all the features that can be extracted without it
                logging.warning('Dependency tree not available')

            # extract sentence-internal n-grams of the right PoS tag
            if self.extract_unigram_features:
                # just unigrams, can get away without sorting the tokens
                for token in parse_tree.nodes_iter():
                    if token.pos not in self.extract_unigram_features:
                        continue
                    features.append(DocumentFeature('1-GRAM', (token,)))

            # some tests use standard bigrams, extract them too
            if self.standard_ngram_features > 1:
                # the tokens are stored as nodes in the parse tree in ANY order, sort them
                sentence = sorted(parse_tree.nodes(), key=attrgetter('index'))
                n_tokens = len(sentence)
                for n in range(2, min(self.standard_ngram_features + 1, n_tokens + 1)):
                    for i in range(n_tokens - n + 1):
                        feature = DocumentFeature('%d-GRAM' % n, tuple(sentence[i: i + n]))
                        features.append(feature)
        # it doesn't matter where in the sentence/document these features were found
        # erase their index
        for feature in features:
            for token in feature.tokens:
                token.index = 'any'

        # remove all features that aren't right- they are there because the code above doesnt
        # put the features through the validation code in DocumentFeature.from_string
        # e.g. the verb phrase "123/V_$$$/N" is not put through validation, so it will be returned as feature
        return [f for f in features if DocumentFeature.from_string(str(f)).type != 'EMPTY']

    def filter_preextracted_features(self, feature_list):
        res = []
        for feat_str in feature_list:
            feat = DocumentFeature.from_string(feat_str)
            if feat.type == 'EMPTY':
                continue
            if feat.type == '1-GRAM' and feat.tokens[0].pos not in self.extract_unigram_features:
                continue
            if feat.type != '1-GRAM' and feat.type not in self.extract_phrase_features:
                continue
            res.append(feat)
        # logging.info('Had %d feats, keeping %d', len(feature_list), len(res))
        return res

    def remove_features_containing_named_entities(self, features):
        return [f for f in features if not any(token.ner in self.entity_ner_tags for token in f.tokens)]
