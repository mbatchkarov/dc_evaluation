# coding=utf-8
from collections import defaultdict
import logging
import gzip
import json

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import networkx as nx
from discoutils.tokens import Token

try:
    import xml.etree.cElementTree as ET
except ImportError:
    logging.warning('cElementTree not available')
    import xml.etree.ElementTree as ET

# copied from feature extraction toolkit
pos_coarsification_map = defaultdict(lambda: "UNK")
pos_coarsification_map.update({"JJ": "J",
                               "JJN": "J",
                               "JJS": "J",
                               "JJR": "J",

                               "VB": "V",
                               "VBD": "V",
                               "VBG": "V",
                               "VBN": "V",
                               "VBP": "V",
                               "VBZ": "V",

                               "NN": "N",
                               "NNS": "N",
                               "NNP": "N",
                               "NPS": "N",
                               "NP": "N",

                               "RB": "RB",
                               "RBR": "RB",
                               "RBS": "RB",

                               "DT": "DET",
                               "WDT": "DET",

                               "IN": "CONJ",
                               "CC": "CONJ",

                               "PRP": "PRON",
                               "PRP$": "PRON",
                               "WP": "PRON",
                               "WP$": "PRON",

                               ".": "PUNCT",
                               ":": "PUNCT",
                               ":": "PUNCT",
                               "": "PUNCT",
                               "'": "PUNCT",
                               "\"": "PUNCT",
                               "'": "PUNCT",
                               "-LRB-": "PUNCT",
                               "-RRB-": "PUNCT",

                               # the four NE types that FET 0.3.6 may return as PoS tags
                               # not really needed in the classification pipeline yet
                               "PERSON": "PERSON",
                               "LOC": "LOC",
                               "ORG": "ORG",
                               "NUMBER": "NUMBER"})


def is_number(s):
    """
    Checks if the given string is an int or a float. Numbers with thousands
    separators (e.g. "1,000.12") are also recognised. Returns true of the string
    contains only digits and punctuation, e.g. 12/23
    """
    try:
        float(s)
        is_float = True
    except ValueError:
        is_float = False

    is_only_digits_or_punct = True
    for ch in s:
        if ch.isalpha():
            is_only_digits_or_punct = False
            break

    return is_float or is_only_digits_or_punct  # or is_int


class BaseTokeniser(object):
    def __init__(self, normalise_entities=False, use_pos=True,
                 coarse_pos=True, lemmatize=True, lowercase=True,
                 remove_stopwords=False, remove_short_words=False,
                 remove_long_words=False,
                 dependency_format='collapsed-ccprocessed'):
        self.normalise_entities = normalise_entities
        self.use_pos = use_pos
        self.coarse_pos = coarse_pos
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.remove_short_words = remove_short_words
        self.remove_long_words = remove_long_words
        self.dependency_format = dependency_format

    def tokenize_corpus(self, *args, **kwargs):
        raise NotImplementedError


class XmlTokenizer(BaseTokeniser):
    def tokenize_corpus(self, file_names, corpus_name):
        logging.info('%s running for %s', self.__class__.__name__, corpus_name)
        # i is needed to get the ID of the doc in case something goes wrong
        trees = []
        for (i, x) in enumerate(file_names):
            with open(x) as infile:
                trees.append(self.tokenize_doc(infile.read()))
        return trees

    def tokenize_doc(self, doc, **kwargs):
        """
        Tokenizes a Stanford Core NLP processed document by parsing the XML and
        extracting tokens and their lemmas, with optional lowercasing
        If requested, the named entities will be replaced with the respective
         type, e.g. PERSON or ORG, otherwise numbers and punctuation will be
         canonicalised

         :returns: a list of sentence tuple of the form (tokens_list,
                                                (dependency_graph, {token index in sentence -> token object})
                                                )
        """
        try:
            # tree = ET.fromstring(doc.encode("utf8"))
            tree = ET.fromstring(doc)
            sentences = []
            for sent_element in tree.findall('.//sentence'):
                sentences.append(self._process_sentence(sent_element))
        except ET.ParseError as e:
            logging.error('Parse error %s', e)
            pass
            # on OSX the .DS_Store file is passed in, if it exists
            # just ignore it
        return sentences

    def _process_sentence(self, tree):
        """
        Build a dependency tree (networkx.DiGraph) for a sentence from its corresponding XML tree
        :param tree:
        :return:
        """
        tokens = []
        for element in tree.findall('.//token'):
            if self.lemmatize:
                txt = element.find('lemma').text
            else:
                txt = element.find('word').text
                # check if the token is a number/stopword before things have been done to it
            am_i_a_number = is_number(txt)

            if self.remove_stopwords and txt.lower() in ENGLISH_STOP_WORDS:
                # logging.debug('Tokenizer ignoring stopword %s' % txt)
                continue

            if self.remove_short_words and len(txt) <= 3:
                continue

            if self.remove_long_words and len(txt) >= 25:
                continue

            pos = element.find('POS').text.upper() if self.use_pos else ''
            if self.coarse_pos:
                pos = pos_coarsification_map[pos.upper()]

            if pos == 'PUNCT' or am_i_a_number:
                # logging.debug('Tokenizer ignoring stopword %s' % txt)
                continue

            try:
                iob_tag = element.find('NER').text.upper()
            except AttributeError:
                # logging.error('You have requested named entity '
                # 'normalisation, but the input data are '
                # 'not annotated for entities')
                iob_tag = 'MISSING'
                # raise ValueError('Data not annotated for named entities')

            if '/' in txt or '_' in txt:
                # I use these chars as separators later, remove them now to avoid problems down the line
                logging.debug('Funny token found: %s, pos is %s', txt, pos)
                continue

            if self.lowercase:
                txt = txt.lower()

            if self.normalise_entities:
                if iob_tag != 'O':
                    txt = '__NER-%s__' % iob_tag
                    pos = ''  # normalised named entities don't need a PoS tag

            tokens.append(Token(txt, pos, int(element.get('id')), ner=iob_tag))

        token_index = {t.index: t for t in tokens}

        # build a graph from the dependency information available in the input
        tokens_ids = set(x.index for x in tokens)
        dep_tree = nx.DiGraph()
        dep_tree.add_nodes_from(tokens)

        dependencies = tree.find('.//{}-dependencies'.format(self.dependency_format))
        # some file are formatted like so: <basic-dependencies> ... </basic-dependencies>
        if not dependencies:
            # and some like so: <dependencies type="basic-dependencies"> ... </dependencies>
            # if one fails try the other. If that fails too something is wrong- perhaps corpus has not been parsed?
            dependencies = tree.find(".//dependencies[@type='{}-dependencies']".format(self.dependency_format))
        if dependencies:
            for dep in dependencies.findall('.//dep'):
                type = dep.get('type')
                head = dep.find('governor')
                head_idx = int(head.get('idx'))

                dependent = dep.find('dependent')
                dependent_idx = int(dependent.get('idx'))
                if dependent_idx in tokens_ids and head_idx in tokens_ids:
                    dep_tree.add_edge(token_index[head_idx], token_index[dependent_idx], type=type)

        return dep_tree

    def __str__(self):
        return 'XmlTokenizer:{}'.format(self.important_params)


class ConllTokenizer(XmlTokenizer):
    def tokenize_doc(self, doc_content, **kwargs):
        sentences, this_sent = [], []
        for line in doc_content.split('\n'):
            if not line.strip():
                sentences.append(self._process_sentence(this_sent))
                this_sent = []
            else:
                this_sent.append(line.strip())
        return sentences

    def _process_sentence(self, lines):
        """
        Build a dependency tree (networkx.DiGraph) for a sentence from its corresponding XML tree
        """
        tokens = []
        dependencies = {}  # (from_idx, to_idx) --> type
        for line in lines:
            idx, txt, lemma, pos, ner, head_idx, dep_type = line.split('\t')
            if self.lemmatize:
                txt = lemma
                # check if the token is a number/stopword before things have been done to it
            am_i_a_number = is_number(txt)

            if self.remove_stopwords and txt.lower() in ENGLISH_STOP_WORDS:
                # logging.debug('Tokenizer ignoring stopword %s' % txt)
                continue

            if self.remove_short_words and len(txt) <= 3:
                continue

            if self.remove_long_words and len(txt) >= 25:
                continue

            pos = pos.upper() if self.use_pos else ''
            if self.coarse_pos:
                pos = pos_coarsification_map[pos.upper()]

            if pos == 'PUNCT' or am_i_a_number:
                # logging.debug('Tokenizer ignoring stopword %s' % txt)
                continue

            iob_tag = ner.upper()

            if '/' in txt or '_' in txt:
                # I use these chars as separators later, remove them now to avoid problems down the line
                logging.debug('Funny token found: %s, pos is %s', txt, pos)
                continue

            if self.lowercase:
                txt = txt.lower()

            if self.normalise_entities:
                if iob_tag != 'O':
                    txt = '__NER-%s__' % iob_tag
                    pos = ''  # normalised named entities don't need a PoS tag

            dependencies[(int(idx), int(head_idx))] = dep_type
            tokens.append(Token(txt, pos, int(idx), ner=iob_tag))

        token_index = {t.index: t for t in tokens}

        # build a graph from the dependency information available in the input
        tokens_ids = set(x.index for x in tokens)
        dep_tree = nx.DiGraph()
        dep_tree.add_nodes_from(tokens)

        for (dep_idx, head_idx), type in dependencies.items():
            if head_idx in tokens_ids and dep_idx in tokens_ids:
                dep_tree.add_edge(token_index[head_idx], token_index[dep_idx],  type=type)
        return dep_tree


# todo json tokenizers probably ignore the short/stopword parameters
class GzippedJsonTokenizer(BaseTokeniser):
    def tokenize_corpus(self, tar_file, *args, **kwargs):
        logging.info('Compressed JSON tokenizer running for %s', tar_file)
        labels, docs = [], []
        with gzip.open(tar_file, 'rb') as infile:
            for line in infile:
                d = json.loads(line.decode('UTF8'))
                labels.append(d[0])
                docs.append(d[1])
        return docs, labels
