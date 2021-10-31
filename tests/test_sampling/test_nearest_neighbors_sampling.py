import math

import pytest

from deduplipy.datasets import load_data
from deduplipy.sampling import NearestNeighborsPairsSampler


@pytest.mark.parametrize('n_samples', [100, 1_000, 10_000])
def test_nn_sampler_base(n_samples):
    df = load_data('voters')
    myNNSampling = NearestNeighborsPairsSampler(col_names=df.columns.tolist(), n_neighbors=10)
    assert set(myNNSampling.pairs_col_names) == {'name_1', 'suburb_1', 'postcode_1', 'name_2', 'suburb_2', 'postcode_2'}
    sample = myNNSampling.sample(df, n_samples=n_samples)
    assert math.isclose(n_samples, len(sample), rel_tol=0.1)
    assert set(sample.columns) == {'name_1', 'suburb_1', 'postcode_1', 'name_2', 'suburb_2', 'postcode_2'}
    assert math.isclose(n_samples, len(sample), rel_tol=0.1)
