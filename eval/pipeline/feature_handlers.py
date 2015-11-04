from discoutils.tokens import DocumentFeature
from discoutils.thesaurus_loader import Thesaurus
from eval.pipeline.thesauri import DummyThesaurus
from eval.utils.reflection_utils import get_named_object


def get_token_handler(handler_name, k, transformer_name, thesaurus):
    """

    :param handler_name: fully qualified name of the handler class. Must implement the BaseFeatureHandler interface
    :param k: if the handler maker replacements, how many neighbours to insert for each replaced feature
    :param transformer_name: fully qualified function name of the function (float -> float) that transforms the
    similarity score between a feature and its replacements.
    :param thesaurus: source of vectors or neighbours used to make replacements
    :return:
    """
    handler = get_named_object(handler_name)
    transformer = get_named_object(transformer_name)
    return handler(k, transformer, thesaurus)


class BaseFeatureHandler():
    """
    Base class for all feature handlers. This is used during document vectorisation and decides what to do with each
    newly encountered document features. Currently the options are:
      - ignore it. This is the standard test-time behaviour of `CountVectorizer` for out-of-vocabulary features.
      - enter it into the vocabulary and increment the corresponding column in the document vector. This is the default
      train-time behaviour of `CountVectorizer`
      - replace it with other features according to a distributional model.

    The decision is based on whether the feature is in the current model vocabulary (IV) or not (OOV), and whether
    it is in the distributional model (IT) or not (OOT).

    This class does standard CountVectorizer-like vectorization:
        - in vocabulary, in thesaurus: only insert feature itself
        - IV,OOT: feature itself
        - OOV, IT: ignore feature
        - OOV, OOT: ignore feature
    """

    def __init__(self, *args):
        pass

    def handle_IV_IT_feature(self, **kwargs):
        self._insert_feature_only(**kwargs)

    def handle_IV_OOT_feature(self, **kwargs):
        self._insert_feature_only(**kwargs)

    def handle_OOV_IT_feature(self, **kwargs):
        self._ignore_feature()

    def handle_OOV_OOT_feature(self, **kwargs):
        self._ignore_feature()

    def _insert_feature_only(self, feature_index_in_vocab, j_indices, values, **kwargs):
        j_indices.append(feature_index_in_vocab)
        values.append(1)

    def _ignore_feature(self):
        pass

    def _paraphrase(self, feature, vocabulary, j_indices, values, stats, **kwargs):
        """
        Replaces term with its k nearest neighbours from the thesaurus

        Parameters
        ----------
        neighbour_source : callable, returns a thesaurus-like object (a list of
          (neighbour, sim) tuples, sorted by highest sim first,
          acts as a defaultdict(list) ). The callable takes one parameter for
          compatibility purposes- one of the possible callables I want to
          use here requires access to the vocabulary.
           The default behaviour is to return a callable pointing to the
           currently loaded thesaurus.
        """

        # logging.debug('Paraphrasing %r in doc %d', feature, doc_id)
        neighbours = self.thesaurus.get_nearest_neighbours(feature)
        if isinstance(self.thesaurus, Thesaurus):
            # precomputed thesauri do not guarantee that the returned neighbours will be in vocabulary
            # these should by now only the used in testing though
            neighbours = [(neighbour, sim) for (neighbour, sim) in neighbours
                          if DocumentFeature.from_string(neighbour) in vocabulary]
        event = [str(feature), len(neighbours)]
        for neighbour, sim in neighbours[:self.k]:
            # the document may already contain the feature we
            # are about to insert into it,
            # a merging strategy is required,
            # e.g. what do we do if the document has the word X
            # in it and we encounter X again. By default,
            # scipy uses addition
            df = DocumentFeature.from_string(neighbour)
            j_indices.append(vocabulary.get(df))
            values.append(self.sim_transformer(sim))
            # track the event
            event.extend([neighbour, sim])
        stats.register_paraphrase(tuple(event))


class SignifierSignifiedFeatureHandler(BaseFeatureHandler):
    """
    Handles features the way standard Naive Bayes does, except
        - OOV, IT: insert the first K IV neighbours from thesaurus instead of
        ignoring the feature
    This is standard feature expansion from the IR literature.
    """

    def __init__(self, k, sim_transformer, thesaurus):
        self.k = k
        self.sim_transformer = sim_transformer
        self.thesaurus = thesaurus

    def handle_OOV_IT_feature(self, **kwargs):
        self._paraphrase(**kwargs)


class SignifiedOnlyFeatureHandler(BaseFeatureHandler):
    """
    Ignores all OOT features and inserts the first K IV neighbours from
    thesaurus for all IT features. This is what I called Extreme Feature Expansion
    in my thesis
    """

    def __init__(self, k, sim_transformer, thesaurus):
        self.k = k
        self.sim_transformer = sim_transformer
        self.thesaurus = thesaurus

    def handle_OOV_IT_feature(self, **kwargs):
        self._paraphrase(**kwargs)

    handle_IV_IT_feature = handle_OOV_IT_feature

    def handle_IV_OOT_feature(self, **kwargs):
        self._ignore_feature()


class SignifierRandomBaselineFeatureHandler(SignifiedOnlyFeatureHandler):
    """
    Ignores all OOT features and inserts K random IV tokens for all IT features. Useful to unit tests.
    """

    def __init__(self, k, sim_transformer, thesaurus):
        self.k = k
        self.sim_transformer = sim_transformer
        self.thesaurus = thesaurus

    def handle_OOV_IT_feature(self, **kwargs):
        self._paraphrase(**kwargs)

    handle_IV_IT_feature = handle_OOV_IT_feature
