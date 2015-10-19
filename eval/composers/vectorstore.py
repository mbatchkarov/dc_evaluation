from random import sample
from discoutils.thesaurus_loader import Thesaurus
from discoutils.tokens import DocumentFeature


def check_vectors(unigram_source):
    if not unigram_source:
        raise ValueError('Composers need a unigram vector source')
    if not hasattr(unigram_source, 'get_vector'):
        raise ValueError('Creating a composer requires a Vectors data structure that holds unigram vectors')
    return unigram_source



class DummyThesaurus(Thesaurus):
    """
    A thesaurus-like object which return "b/N" as the only neighbour of every possible entry
    """
    name = 'Constant'

    def __init__(self):
        pass

    def get_nearest_neighbours(self, feature):
        return [('b/N', 1.0)]

    def get_vector(self):
        pass

    def to_shelf(self, *args, **kwargs):
        pass

    def __len__(self):
        return 9999999

    def __contains__(self, feature):
        return True


class RandomThesaurus(DummyThesaurus):
    """
    A thesaurus-like object which returns a single random neighbour for every possible entry. That neighbour
    is chosen from the vocabulary that is passed in (as a dict {feature:index} )
    """
    name = 'Random'

    def __init__(self, vocab=None, k=1):
        self.vocab = vocab
        self.k = k

    def get_nearest_neighbours(self, item):
        if not self.vocab:
            raise ValueError('You need to provide a set of value to choose from first.')
        return [(str(foo), 1.) for foo in sample(self.vocab, self.k)]


def _default_row_filter(feat_str: str, feat_df: DocumentFeature):
    return feat_df.tokens[0].pos in {'N', 'J', 'V'} and feat_df.type == '1-GRAM'

