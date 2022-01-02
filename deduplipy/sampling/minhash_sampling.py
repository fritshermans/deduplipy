from typing import List, Tuple

import numpy as np
import pandas as pd
from pyminhash import MinHash

from .sampler import Sampler


class MinHashSampler(Sampler):
    """
    Class to create a pairs table sample for `col_names` by applying minhashing with `n_hash_tables` hash tables.
    The Scikit-Learn `CountVectorizer` is used for tokenization.

    Args:
        col_names: column names to use for creating pairs
        n_hash_tables: number of hash tables to use for hashing
        analyzer: way how CountVectorizer creates tokens
        ngram_range: range of n-grams sizes the CountVectorizer uses

    """
    def __init__(self, col_names: List[str], n_hash_tables=10, ngram_range: Tuple[int] = (1, 1),
                 analyzer: str = 'word'):
        super().__init__(col_names)
        self.n_hash_tables = n_hash_tables
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.MinHasher = MinHash(self.n_hash_tables, ngram_range=self.ngram_range, analyzer=self.analyzer)

    def _create_minhash_pairs(self, X: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """
        Create pairs of rows based on minhashing. Only pairs with a Jaccard similarity larger than `threshold` wil be
        included. When multiple columns are used for minhashing, the mean of their Jaccard similarities per pair is
        calculated for thresholding.

        Args:
            X: Pandas dataframe
            threshold: Jaccard similarity threshold

        Returns:
            Pandas dataframe containing pairs

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
        return minhash_pairs

    def _get_stratified_sample(self, minhash_pairs: pd.DataFrame, n_samples: int, n_buckets: int = 10) -> pd.DataFrame:
        """
        Create a stratified sample per Jaccard similarity bucket.

        Args:
            minhash_pairs: Pandas dataframe containing minhash pair results
            n_samples: number of samples that need to be generated
            n_buckets: number of buckets to be applied for stratified sampling

        Returns:
            Pandas dataframe with stratified sample

        """
        minhash_pairs['distance_bucket'] = pd.cut(minhash_pairs['jaccard_sim'], bins=n_buckets)

        stratified_sample = minhash_pairs.groupby('distance_bucket', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), n_samples // n_buckets), replace=False))
        return stratified_sample

    def _get_non_stratified_sample(self, minhash_pairs: pd.DataFrame, stratified_sample: pd.DataFrame,
                                   n_samples: int) -> pd.DataFrame:
        """
        Create sample to be added to stratified sample to get a total of `n_samples`

        Args:
            minhash_pairs: Pandas dataframe containing minhash pair results
            stratified_sample: Pandas dataframe containing stratified sample results
            n_samples: total number of samples required

        Returns:
            Pandas dataframe containing non-stratified sample of minhash pairs

        """
        n_stratified_sample = len(stratified_sample)
        n_non_stratified_sample = n_samples - n_stratified_sample

        stratified_sample['stratified'] = True
        non_stratified_sample = (minhash_pairs.merge(stratified_sample, how='left', on=minhash_pairs.columns.tolist())
                                 .fillna({'stratified': False}))
        non_stratified_sample = (non_stratified_sample[~non_stratified_sample['stratified']]
                                     .sample(frac=1)
                                     .iloc[:n_non_stratified_sample])
        return non_stratified_sample

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
        minhash_pairs = self._create_minhash_pairs(X, threshold)

        stratified_sample = self._get_stratified_sample(minhash_pairs, n_samples)

        non_stratified_sample = self._get_non_stratified_sample(minhash_pairs, stratified_sample, n_samples)

        sample = stratified_sample.append(non_stratified_sample)[self.pairs_col_names]

        return sample
