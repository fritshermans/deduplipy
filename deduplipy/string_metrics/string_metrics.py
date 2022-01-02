import numpy as np

from thefuzz.fuzz import ratio, token_sort_ratio, token_set_ratio, partial_ratio


def length_adjustment(x_1: str, x_2: str) -> float:
    """
    Factor to adjust for cases where at least one of the pairs is short. This factor is to be applied to string
    similarity metrics.

    Args:
        x_1: string to compare
        x_2: string to compare

    Returns:
        adjustment factor
    """
    shortest_length = min(len(x_1), len(x_2))
    return 1 - np.exp(-0.2 * shortest_length)


def adjusted_ratio(x_1: str, x_2: str) -> float:
    """
    Length adjusted version of `ratio` metric. When at least one of the strings is short, the metric is reduced.

    Args:
        x_1: string to compare
        x_2: string to compare

    Returns:
        adjusted `ratio` metric
    """
    return length_adjustment(x_1, x_2) * ratio(x_1, x_2)


def adjusted_token_sort_ratio(x_1: str, x_2: str) -> float:
    """
    Length adjusted version of `token_sort_ratio` metric. When at least one of the strings is short, the metric is
    reduced.

    Args:
        x_1: string to compare
        x_2: string to compare

    Returns:
        adjusted `token_sort_ratio` metric
    """
    return length_adjustment(x_1, x_2) * token_sort_ratio(x_1, x_2)


def adjusted_token_set_ratio(x_1: str, x_2: str) -> float:
    """
    Length adjusted version of `token_set_ratio` metric. When at least one of the strings is short, the metric is
    reduced.

    Args:
        x_1: string to compare
        x_2: string to compare

    Returns:
        adjusted `token_set_ratio` metric
    """
    return length_adjustment(x_1, x_2) * token_set_ratio(x_1, x_2)


def adjusted_partial_ratio(x_1: str, x_2: str) -> float:
    """
    Length adjusted version of `partial_ratio` metric. When at least one of the strings is short, the metric is reduced.

    Args:
        x_1: string to compare
        x_2: string to compare

    Returns:
        adjusted `partial_ratio` metric
    """
    return length_adjustment(x_1, x_2) * partial_ratio(x_1, x_2)
