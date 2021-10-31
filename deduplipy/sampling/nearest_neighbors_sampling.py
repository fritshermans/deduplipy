from typing import List, Tuple

import pandas as pd
import numpy as np

from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

from .sampling import Sampling


class NearestNeighborsPairsSampler(Sampling):
    def __init__(self, col_names: List[str], n_neighbors: int = 2, metric: str = 'manhattan', analyzer: str = 'char_wb',
                 ngram_range: Tuple[int] = (1, 5)):
        super().__init__(col_names)
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.cv = CountVectorizer(analyzer=self.analyzer, ngram_range=self.ngram_range)
        self.pipe = make_column_transformer(*[(self.cv, col) for col in self.col_names])
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)

    def _calculate_nearest_neighbors(self, X: pd.DataFrame) -> Tuple[np.array, np.array]:
        vectors = self.pipe.fit_transform(X)
        self.nn.fit(vectors)
        distance, index = self.nn.kneighbors(vectors)
        return distance, index

    def sample(self, X: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        distance, index = self._calculate_nearest_neighbors(X)

        pairs = X.copy()
        pairs['distance'] = distance[:, 1]
        pairs['index_2'] = index[:, 1]
        pairs_filtered = pairs.reset_index()
        pairs_filtered = pairs_filtered[pairs_filtered['index'] < pairs_filtered['index_2']]
        pairs = (pairs.merge(pairs_filtered[self.col_names + ['index']],
                             left_on='index_2', right_on='index', suffixes=('_1', '_2'), how='inner'))
        pairs['distance_bucket'] = pd.cut(pairs['distance'], bins=10)

        sample = pairs.groupby('distance_bucket', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), n_samples // 10), replace=False))

        return sample[self.pairs_col_names]
