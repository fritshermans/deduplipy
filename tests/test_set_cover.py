from deduplipy.blocking.set_cover import greedy_set_cover


def test_greedy_set_cover_base_case():
    subsets = [[0, 1, 2, 3], [1, 2], [1, 4]]
    parent_set = {0, 1, 2, 3, 4}
    result = greedy_set_cover(subsets, parent_set)
    assert result == [{0, 1, 2, 3}, {1, 4}]


def test_greedy_set_cover_recall():
    subsets = [[0, 1, 2, 3], [1, 2], [1, 4]]
    parent_set = {0, 1, 2, 3, 4}
    result = greedy_set_cover(subsets, parent_set, recall=0.8)
    assert result == [{0, 1, 2, 3}]
