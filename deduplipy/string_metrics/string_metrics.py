import numpy as np

from fuzzywuzzy.fuzz import ratio, token_sort_ratio, token_set_ratio, partial_ratio


def length_adjustment(x_1, x_2):
    shortest_length = min(len(x_1), len(x_2))
    return 1 - np.exp(-0.2 * shortest_length)


def adjusted_ratio(x_1, x_2):
    return length_adjustment(x_1, x_2) * ratio(x_1, x_2)


def adjusted_token_sort_ratio(x_1, x_2):
    return length_adjustment(x_1, x_2) * token_sort_ratio(x_1, x_2)


def adjusted_token_set_ratio(x_1, x_2):
    return length_adjustment(x_1, x_2) * token_set_ratio(x_1, x_2)


def adjusted_partial_ratio(x_1, x_2):
    return length_adjustment(x_1, x_2) * partial_ratio(x_1, x_2)
