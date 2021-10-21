import numpy as np

from fancyimpute import SoftImpute

from deduplipy.config import CONVERGENCE_THRESHOLD


def fill_missing_links(matrix, convergence_threshold=CONVERGENCE_THRESHOLD):
    """
    Fill missing values in adjacency matrix using SoftImpute. Missing values are considered to be zero,
    as this is the default of the `nx.to_numpy_matrix` function when there is no edge between two nodes.

    Args:
        matrix: adjacency matrix
        convergence_threshold: convergence threshold for SoftImpute algorithm

    Returns:
        Numpy adjacency matrix with imputed missing values

    """
    matrix_ = matrix.copy()
    np.fill_diagonal(matrix_, 1)
    matrix_[matrix_ == 0] = np.nan
    if not np.isnan(matrix_).any():
        return matrix

    imputer = SoftImpute(min_value=0, max_value=1, verbose=False, convergence_threshold=convergence_threshold,
                         init_fill_method='mean')  # init_fill_method='mean' significantly improves speed
    matrix_ = imputer.fit_transform(matrix_)
    # the adjacency matrix needs to have zeros on the diagonal
    np.fill_diagonal(matrix_, 0)

    # force symmetry
    matrix_ = np.tril(matrix_) + np.triu(matrix_.T, 1)
    return matrix_
