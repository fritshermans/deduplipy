import numpy as np

from scipy.optimize import differential_evolution

from fancyimpute import SoftImpute


def _get_missing_positions(matrix):
    pos_missing = []
    rows, cols = np.triu_indices_from(matrix)
    for row, col in zip(rows, cols):
        if matrix[row, col] == 0:
            pos_missing.append([row, col])
    return pos_missing


def fill_missing_links_diff_evol(matrix):
    matrix_ = matrix.copy()
    # we assume a correlation matrix:
    np.fill_diagonal(matrix_, 1)

    def helper_function(x):
        for val, pos in zip(x, pos_missing):
            matrix_[pos[0], pos[1]] = val
            matrix_[pos[1], pos[0]] = val
        eigenvalues = np.linalg.eigvals(matrix_)

        return -min(eigenvalues)

    pos_missing = _get_missing_positions(matrix)
    if not len(pos_missing):
        return matrix

    bounds = [(0, 1) for _ in pos_missing]

    res = differential_evolution(helper_function, bounds)

    for val, pos in zip(res.x, pos_missing):
        matrix_[pos[0], pos[1]] = val
        matrix_[pos[1], pos[0]] = val

    # the adjacency matrix needs to have zeros on the diagonal
    np.fill_diagonal(matrix_, 0)

    return matrix_


def fill_missing_links(matrix):
    matrix_ = matrix.copy()
    matrix_[matrix_ == 0] = np.nan
    np.fill_diagonal(matrix_, 1)

    imputer = SoftImpute(min_value=0, max_value=1, verbose=False)
    matrix_ = imputer.complete(matrix_)
    np.fill_diagonal(matrix_, 0)

    # force symmetry
    matrix_ = np.tril(matrix_) + np.triu(matrix_.T, 1)
    return matrix_
