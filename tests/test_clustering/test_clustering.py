import os

import pandas as pd

from deduplipy.clustering.clustering import hierarchical_clustering


def test_hierarchical_clustering_base():
    col_names = ['name']
    scored_pairs_table = pd.read_csv(os.path.join('tests', 'test_clustering', 'clustering_fixture.csv'))
    res = hierarchical_clustering(scored_pairs_table, col_names)
    expected = pd.DataFrame({'deduplication_id': [1, 1, 2, 3, 4],
                             'row_number': [0, 1, 2, 3, 4]})
    pd.testing.assert_frame_equal(res, expected)
    assert True


def test_hierarchical_clustering_cluster_threshold():
    col_names = ['name']
    scored_pairs_table = pd.read_csv(os.path.join('tests', 'test_clustering', 'clustering_fixture.csv'))
    res = hierarchical_clustering(scored_pairs_table, col_names, cluster_threshold=0.4)
    expected = pd.DataFrame({'deduplication_id': [1, 1, 2, 2, 3],
                             'row_number': [0, 1, 2, 3, 4]})
    pd.testing.assert_frame_equal(res, expected)
    assert True


def test_hierarchical_clustering_inspect():
    col_names = ['name']
    scored_pairs_table = pd.read_csv(os.path.join('tests', 'test_clustering', 'clustering_fixture.csv'))
    _ = hierarchical_clustering(scored_pairs_table, col_names, inspect={'name': 'frits'})
    assert os.path.exists('dendrogram.png')
    assert os.path.exists('network.png')
    assert True
