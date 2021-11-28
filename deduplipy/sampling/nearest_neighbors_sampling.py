from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

from .sampling import Sampling


class NearestNeighborsPairsSampler(Sampling):
    def __init__(self, col_names: List[str], n_neighbors: int = 2, metric: str = 'manhattan', analyzer: str = 'char_wb',
                 ngram_range: Tuple[int] = (3, 3)):
        """
        Class to create a pairs table sample for `col_names` by taking `n_neighbors` nearest neighbors in the vector
        space created by a ScikitLearn CountVectorizer with `analyzer`, `metric` and `ngram_range`.

        Args:
            col_names: column names to use for creating pairs
            n_neighbors: number of nearest neighbors to include
            metric: distance metric to use for the CountVectorizer
            analyzer: way how CountVectorizer creates tokens
            ngram_range: range of n-grams sizes the CountVectorizer uses

        """
        super().__init__(col_names)
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.cv = CountVectorizer(analyzer=self.analyzer, ngram_range=self.ngram_range, max_features=1_000, min_df=2)
        self.pipe = make_column_transformer(*[(self.cv, col) for col in self.col_names])
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)

    def _calculate_nearest_neighbors(self, X: pd.DataFrame) -> Tuple[np.array, np.array]:
        """
        Method to find nearest neighbors using the pipeline containing a CountVectorizer and NearestNeighbors for
        Pandas dataframe X.

        Args:
            X: Pandas dataframe for which nearest neighbors should be found

        Returns:
            tuple containing distances and indices of nearest neighbors

        Note: The first distance is distance of a point with itself and the first column is therefore always 0. Same
        applies to the indices; the first column is always the row number of the row itself.

        """
        vectors = self.pipe.fit_transform(X)
        self.nn.fit(vectors)
        distance, index = self.nn.kneighbors(vectors)
        return distance, index

    def sample(self, X: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """
        Method to draw sample of pairs of size `n_samples` from dataframe X. Note that `n_samples` cannot be returned if
        the number of nearest neighbors of the object instance is too low.

        Args:
            X: Pandas dataframe containing records to create a sample of pairs from
            n_samples: number of samples to create

        Returns:
            Pandas dataframe containing the sampled pairs

        """

        distance, index = self._calculate_nearest_neighbors(X)

        pairs = X.copy()
        pairs['distance'] = distance[:, 1:].tolist()
        pairs['index_2'] = index[:, 1:].tolist()
        pairs['distance_index_2'] = pairs.apply(lambda row: list(zip(row['distance'], row['index_2'])), axis=1)
        pairs = pairs.explode('distance_index_2').drop(columns=['distance', 'index_2'])
        pairs[['distance', 'index_2']] = pairs['distance_index_2'].tolist()
        pairs = pairs.drop(columns=['distance_index_2'])

        pairs_filtered = pairs.reset_index()
        pairs_filtered = pairs_filtered[pairs_filtered['index'] < pairs_filtered['index_2']]
        pairs = (pairs.merge(pairs_filtered[self.col_names + ['index']],
                             left_on='index_2', right_on='index', suffixes=('_1', '_2'), how='inner'))
        pairs['distance_bucket'] = pd.cut(pairs['distance'], bins=10)

        sample = pairs.groupby('distance_bucket', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), n_samples // 10), replace=False))

        return sample[self.pairs_col_names]
