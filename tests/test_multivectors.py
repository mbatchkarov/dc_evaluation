import pytest

from discoutils.thesaurus_loader import Vectors
from eval.pipeline.multivectors import MultiVectors
from tests.test_vector_sources import ones_vectors

def test_multivectors(ones_vectors):
    assert len(ones_vectors) == 4

    mv = MultiVectors([ones_vectors] * 3)
    assert 'a/N' in mv
    assert 'a/NP' not in mv

    for entry in ones_vectors.keys():
        neigh = mv.get_nearest_neighbours(entry)
        n1 = [foo[0] for foo in ones_vectors.get_nearest_neighbours(entry)]
        n2 = [foo[0] for foo in neigh]
        assert n1 == n2

        sims = [foo[1] for foo in neigh]
        assert sims == [1, 1 / 2, 1 / 3]
    assert mv.get_nearest_neighbours('asdf/N') == []


@pytest.mark.parametrize("call_init", [True, False])
def test_all_neighbours_overlap(call_init):
    FEATURE = 'daily/J_pais/N'
    v = Vectors.from_tsv('tests/resources/only_overlapping.txt', allow_lexical_overlap=False)
    mv = MultiVectors([v] * 3)
    if call_init:
        mv.init_sims()
    assert FEATURE in v
    assert FEATURE in mv  # feature is contained in vector set, but...
    # when we look up its neighbours, they all overlap, so nothing is left
    assert mv.get_nearest_neighbours(FEATURE) == []
    assert mv.get_nearest_neighbours('asdf') == []

    assert mv.get_nearest_neighbours('pais/N') is not None
