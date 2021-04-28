from itertools import product

import numpy as np
import pandas as pd

from deduplipy.clustering.clustering import hierarchical_clustering
from deduplipy.active_learning.active_learning import ActiveStringMatchLearner
from deduplipy.blocking.blocking import Blocking


class Deduplicator:
    def __init__(self, col_name, n_queries=999, rules=None, recall=1.0, cache_tables=False):
        """
        Deduplicate entries in Pandas dataframe using column with name `col_name`. Training takes place during a short
        interactive session (interactive learning).

        Usage:
            df = ...
            myDedupliPy = Deduplicator('name_address')
            myDedupliPy.fit(df)
            myDedupliPy.predict(df_train)

        The result is a dataframe with a new column `cluster_id`. Rows with the same `cluster_id` are deduplicated.

        Args:
            recall:
            col_name: name of column to use for deduplication
            n_queries: max number of queries to do during active learning, early stopping will be advised when
                        classifier converged
            rules: list of rules to use for blocking, if not provided, all default rules will be used
            cache_tables: whether to save intermediate results in csv files for analysis

        """
        self.col_name = col_name
        self.n_queries = n_queries
        self.rules = rules
        self.recall = recall
        self.cache_tables = cache_tables
        self.myActiveLearner = ActiveStringMatchLearner(n_queries=self.n_queries, col_name=self.col_name)
        self.myBlocker = Blocking(self.col_name, rules, cache_tables=self.cache_tables, recall=self.recall)

    def _create_pairs_table(self, X, n_samples):
        """
        Create sample of pairs

        Args:
            X: Pandas dataframe containing data to be deduplicated
            n_samples: number of sample pairs to be created

        Returns:
            Pandas dataframe containing pairs

        """
        X_pool = X.copy()
        X_pool['row_number'] = np.arange(len(X_pool))
        df_sample = X_pool.sample(n=int(n_samples ** 0.5))

        sample_combinations = pd.DataFrame(
            list(product(df_sample[[self.col_name, 'row_number']].values.tolist(),
                         df_sample[[self.col_name, 'row_number']].values.tolist())),
            columns=[f'{self.col_name}_1', f'{self.col_name}_2'])

        for nr in [1, 2]:
            sample_combinations[f'row_number_{nr}'] = sample_combinations[f'{self.col_name}_{nr}'].str[1]
            sample_combinations[f'{self.col_name}_{nr}'] = sample_combinations[f'{self.col_name}_{nr}'].str[0]

        sample_combinations.sort_values(['row_number_1', 'row_number_2'], inplace=True)

        sample_combinations = sample_combinations[
            sample_combinations['row_number_1'] <= sample_combinations['row_number_2']]
        sample_combinations = sample_combinations[[f'{self.col_name}_1', f'{self.col_name}_2']].reset_index(drop=True)
        return sample_combinations

    def fit(self, X, n_samples=1_000):
        """
        Fit the deduplicator instance

        Args:
            X: Pandas dataframe to be used for fitting
            n_samples: number of pairs to be created for active learning

        Returns: trained deduplicator instance

        """
        sample_combinations = self._create_pairs_table(X, n_samples)
        self.myActiveLearner.fit(sample_combinations)
        print('active learning finished')
        self.myBlocker.fit(self.myActiveLearner.learner.X_training, self.myActiveLearner.learner.y_training)
        print('blocking rules found')
        print(self.myBlocker.rules_selected)
        return self

    def predict(self, X, score_threshold=0.1):
        """
        Predict on new data using the trained deduplicator.

        Args:
            X: Pandas dataframe with column as used when fitting deduplicator instance
            score_threshold: Classification threshold to use for filtering before starting hierarchical clustering

        Returns: Pandas dataframe with a new column `cluster_id`. Rows with the same `cluster_id` are deduplicated.

        """
        print('blocking started')
        scored_pairs_table = self.myBlocker.transform(X)
        print('blocking finished')
        print(f'Nr of pairs: {len(scored_pairs_table)}')
        print('scoring started')
        scored_pairs_table['score'] = self.myActiveLearner.predict_proba(
            scored_pairs_table[[f'{self.col_name}_1', f'{self.col_name}_2']])[:, 1]
        scored_pairs_table.loc[
            scored_pairs_table[f'{self.col_name}_1'] == scored_pairs_table[f'{self.col_name}_2'], 'score'] = 1
        print("scoring finished")
        scored_pairs_table = scored_pairs_table[scored_pairs_table['score'] >= score_threshold]
        print(f'Nr of filtered pairs: {len(scored_pairs_table)}')
        if self.cache_tables:
            scored_pairs_table.to_csv('scored_pairs_table.csv')
        print('Clustering started')
        df_clusters = hierarchical_clustering(scored_pairs_table, col_name=self.col_name)
        print('Clustering finished')
        return df_clusters
