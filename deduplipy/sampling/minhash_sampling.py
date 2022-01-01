from typing import List, Tuple

import numpy as np
import pandas as pd
from pyminhash import MinHash

from .sampling import Sampling


class MinHashSampler(Sampling):
    def __init__(self, col_names: List[str], n_hash_tables=10, ngram_range: Tuple[int] = (1, 1),
                 analyzer: str = 'word'):
        """
        Class to create a pairs table sample for `col_names` by applying minhashing with `n_hash_tables` hash tables.
        The
        Scikit-Learn `CountVectorizer` is used for tokenization.

        Args:
            col_names: column names to use for creating pairs
            n_hash_tables: number of hash tables to use for hashing
            analyzer: way how CountVectorizer creates tokens
            ngram_range: range of n-grams sizes the CountVectorizer uses

        """
        super().__init__(col_names)
        self.n_hash_tables = n_hash_tables
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.MinHasher = MinHash(self.n_hash_tables, ngram_range=self.ngram_range, analyzer=self.analyzer)

    def sample(self, X: pd.DataFrame, n_samples: int, threshold: float = 0.2) -> pd.DataFrame:
        """
        Method to draw sample of pairs of size `n_samples` from dataframe X. Note that `n_samples` cannot be returned if
        the number of pairs above the threshold is too low.

        Args:
            X: Pandas dataframe containing records to create a sample of pairs from
            n_samples: number of samples to create
            threshold: Jaccard threshold for pair inclusion

        Returns:
            Pandas dataframe containing the sampled pairs

        """
        df = X.copy()
        df['row_number'] = np.arange(len(df))

        minhash_pairs = pd.DataFrame()
        for col in self.col_names:
            minhash_result = self.MinHasher.fit_predict(df, col)

            # add other columns than the one used for minhashing
            minhash_result = (minhash_result
                              .merge(df.drop(columns=[col]), left_on='row_number_1', right_on='row_number')
                              .drop(columns=['row_number']))
            minhash_result = (minhash_result
                              .merge(df.drop(columns=[col]), left_on='row_number_2', right_on='row_number',
                                     suffixes=("_1", "_2"))
                              .drop(columns=['row_number']))
            minhash_pairs = minhash_pairs.append(minhash_result, ignore_index=True)

        # mean of Jaccard similarities over all columns is used for thresholding
        minhash_pairs = (minhash_pairs
                         .groupby(['row_number_1', 'row_number_2'] + self.pairs_col_names, as_index=False)
                         ['jaccard_sim'].mean()
                         .drop(columns=['row_number_1', 'row_number_2']))
        minhash_pairs = minhash_pairs[minhash_pairs['jaccard_sim'] >= threshold]

        return minhash_pairs.sample(frac=1).iloc[:n_samples]
