import pytest
import numpy as np

from deduplipy.string_metrics.string_metrics import (length_adjustment, adjusted_ratio, adjusted_token_sort_ratio,
                                                     adjusted_token_set_ratio, adjusted_partial_ratio)


def test_length_adjustment():
    assert length_adjustment('', '') == 0
    assert length_adjustment('', 'aaaaaaaaaaaaaaaa') == 0
    assert length_adjustment('aaaaaaaaaaaaaaaa', '') == 0
    np.testing.assert_approx_equal(length_adjustment('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                                                     'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'), 1, significant=2)


@pytest.mark.parametrize('string_metric',
                         [adjusted_ratio, adjusted_token_sort_ratio, adjusted_token_set_ratio, adjusted_partial_ratio])
def test_adjusted_ratio(string_metric):
    assert string_metric('', '') == 0
    assert string_metric('', 'aaaaaaaaaaaaaaaa') == 0
    assert string_metric('aaaaaaaaaaaaaaaa', '') == 0
    np.testing.assert_approx_equal(string_metric('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                                                 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'), 100, significant=2)
