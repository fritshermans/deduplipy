import numpy as np

from deduplipy.clustering.fill_missing_edges import fill_missing_links
from scipy.stats import random_correlation


def get_random_matrix_missing_values(size, fraction):
    eigenvalues = np.arange(1, 1 + size)
    eigenvalues = eigenvalues / sum(eigenvalues)
    eigenvalues = eigenvalues * len(eigenvalues)
    matrix = random_correlation.rvs(eigenvalues)

    matrix_missing = matrix.copy()
    mask = np.random.randint(0, 100, size=(size, size)) > 100 * (1 - fraction)
    matrix_missing[mask] = 0
    matrix_missing = np.triu(matrix_missing) + np.triu(matrix_missing).T
    np.fill_diagonal(matrix_missing, 1)
    np.fill_diagonal(matrix, 0)
    return matrix, matrix_missing


def test_fill_missing_links_base():
    matrix = np.asarray([[0, 0.8, 0],
                         [0.8, 0, 0.8],
                         [0, 0.8, 0]])
    res = fill_missing_links(matrix, convergence_threshold=0.01)
    expected = np.array([[0., 0.8, 0.9],
                         [0.8, 0., 0.8],
                         [0.9, 0.8, 0.]])
    np.testing.assert_almost_equal(res, expected, decimal=2)


def test_fill_missing_links_no_missing():
    matrix = np.asarray([[0, 0.8, 0.8],
                         [0.8, 0, 0.8],
                         [0.8, 0.8, 0]])
    res = fill_missing_links(matrix, convergence_threshold=0.01)
    expected = np.array([[0, 0.8, 0.8],
                         [0.8, 0, 0.8],
                         [0.8, 0.8, 0]])
    np.testing.assert_almost_equal(res, expected, decimal=2)


def test_fill_missing_links_large_matrix():
    matrix, matrix_missing = get_random_matrix_missing_values(500, 0.5)
    res = fill_missing_links(matrix, convergence_threshold=0.01)
    np.testing.assert_almost_equal(res, matrix, decimal=2)
