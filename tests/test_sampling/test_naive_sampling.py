import math

import pytest

from deduplipy.datasets import load_data
from deduplipy.sampling import NaiveSampling


@pytest.mark.parametrize('n_samples', [100, 1_000, 10_000])
@pytest.mark.parametrize('n_perfect_matches', [1, 3, 5])
def test_naivesampling_base(n_samples, n_perfect_matches):
    df = load_data('voters')
    myNaiveSampling = NaiveSampling(col_names=df.columns.tolist(), n_perfect_matches=n_perfect_matches)
    assert set(myNaiveSampling.pairs_col_names) == {'name_1', 'suburb_1', 'postcode_1', 'name_2', 'suburb_2',
                                                    'postcode_2'}
    sample = myNaiveSampling.sample(df, n_samples=n_samples)
    assert set(sample.columns) == {'name_1', 'suburb_1', 'postcode_1', 'name_2', 'suburb_2', 'postcode_2'}
    assert math.isclose(n_samples, len(sample), rel_tol=0.1)
    # assert that the first `n_perfect_matches` are actually perfect matches:
    perfect_matches = sample.iloc[:n_perfect_matches]
    assert all([all(perfect_matches[f'{col}_1'] == perfect_matches[f'{col}_2']) for col in df.columns.tolist()])