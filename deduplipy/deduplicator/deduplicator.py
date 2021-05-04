from itertools import product

import numpy as np
import pandas as pd
from fuzzywuzzy.fuzz import ratio, token_sort_ratio

from deduplipy.active_learning.active_learning import ActiveStringMatchLearner
from deduplipy.blocking.blocking import Blocking
from deduplipy.clustering.clustering import hierarchical_clustering
from deduplipy.config import DEDUPLICATION_ID_NAME, ROW_ID


class Deduplicator:
    def __init__(self, col_names=None, field_info=None, interaction=False, n_queries=999, rules=None, recall=1.0,
                 save_intermediate_steps=False, verbose=0):
        """
        Deduplicate entries in Pandas dataframe using columns with names `col_names`. Training takes place during a
        short, interactive session (interactive learning).

        Usage:
            df = ...
            myDedupliPy = Deduplicator(['name', 'address'])
            myDedupliPy.fit(df)
            myDedupliPy.predict(df)

        The result is a dataframe with a new column `deduplication_id`. Rows with the same `deduplication_id` are
        deduplicated.

        Args:
            col_names: list of column names to be used for deduplication, if `col_names` is provided, `field_info` can
            be set to `None` as it will be neglected
            field_info: dict containing column names as keys and lists of metrics per column name as values, only used
            when col_names is `None`
            interaction: whether to include interaction features
            n_queries: max number of queries to do during active learning, early stopping will be advised when
                        classifier converged
            rules: list of rules to use for blocking, if not provided, all default rules will be used
            recall: desired recall reached by blocking rules
            save_intermediate_steps: whether to save intermediate results in csv files for analysis
            verbose: sets verbosity

        """
        if col_names:
            self.col_names = col_names
            self.field_info = {col_name: [ratio, token_sort_ratio] for col_name in
                               self.col_names}
        else:
            self.field_info = field_info
            self.col_names = list(self.field_info.keys())
        self.interaction = interaction
        self.n_queries = n_queries
        self.rules = rules
        self.recall = recall
        self.save_intermediate_steps = save_intermediate_steps
        self.verbose = verbose
        self.myActiveLearner = ActiveStringMatchLearner(n_queries=self.n_queries, col_names=self.col_names,
                                                        interaction=self.interaction, verbose=self.verbose)
        self.myBlocker = Blocking(self.col_names, rules, recall=self.recall,
                                  save_intermediate_steps=self.save_intermediate_steps)

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
        X_pool[ROW_ID] = np.arange(len(X_pool))
        df_sample = X_pool.sample(n=int(n_samples ** 0.5))

        pairs_table = pd.DataFrame(
            list(product(df_sample[self.col_names + [ROW_ID]].values.tolist(),
                         df_sample[self.col_names + [ROW_ID]].values.tolist())))

        pairs_table[[f'{x}_1' for x in self.col_names + [ROW_ID]]] = pairs_table[0].to_list()
        pairs_table[[f'{x}_2' for x in self.col_names + [ROW_ID]]] = pairs_table[1].to_list()
        pairs_table.drop(columns=[0, 1], inplace=True)

        pairs_table.sort_values([f'{ROW_ID}_1', f'{ROW_ID}_2'], inplace=True)

        pairs_table = pairs_table[
            pairs_table[f'{ROW_ID}_1'] <= pairs_table[f'{ROW_ID}_2']]
        self.pairs_col_names = [f'{x}_1' for x in self.col_names] + [f'{x}_2' for x in self.col_names]
        pairs_table = pairs_table[self.pairs_col_names].reset_index(
            drop=True)
        return pairs_table

    def _calculate_string_similarities(self, X):
        metrics_col_names = []
        for field in self.field_info.keys():
            for metric in self.field_info[field]:
                metrics_col_name = f'{field}_{metric.__name__}'
                X[metrics_col_name] = X.apply(lambda row: metric(row[f'{field}_1'], row[f'{field}_2']), axis=1)
                metrics_col_names.append(metrics_col_name)
        X['similarities'] = X.apply(lambda row: [row[metrics_col_name] for metrics_col_name in metrics_col_names],
                                    axis=1)
        X.drop(columns=metrics_col_names, inplace=True)
        return X

    def fit(self, X, n_samples=10_000):
        """
        Fit the deduplicator instance

        Args:
            X: Pandas dataframe to be used for fitting
            n_samples: number of pairs to be created for active learning

        Returns: trained deduplicator instance

        """
        pairs_table = self._create_pairs_table(X, n_samples)
        similarities = self._calculate_string_similarities(pairs_table)
        self.myActiveLearner.fit(similarities)
        if self.verbose:
            print('active learning finished')
        # calculate predictions on pairs table to use for blocking
        y_pred = self.myActiveLearner.predict(similarities['similarities'].to_list())
        self.myBlocker.fit(similarities[self.pairs_col_names], y_pred)
        if self.verbose:
            print('blocking rules found')
            print([x[1:] for x in self.myBlocker.rules_selected])
        return self

    @staticmethod
    def _add_singletons(X):
        """
        Adds `deduplication_id` to rows that are not deduplicated with other rows.

        Args:
            X: deduplication result where singletons have missing values for `deduplication_id`

        Returns:
            deduplication result where singletons have values for `deduplication_id`

        """
        n_missing = len(X[X[DEDUPLICATION_ID_NAME].isnull()])
        max_cluster_id = X[X[DEDUPLICATION_ID_NAME].notnull()][DEDUPLICATION_ID_NAME].max()
        X.loc[X[DEDUPLICATION_ID_NAME].isnull(), DEDUPLICATION_ID_NAME] = np.arange(max_cluster_id + 1,
                                                                                    max_cluster_id + 1 + n_missing)
        return X

    def predict(self, X, score_threshold=0.1):
        """
        Predict on new data using the trained deduplicator.

        Args:
            X: Pandas dataframe with column as used when fitting deduplicator instance
            score_threshold: Classification threshold to use for filtering before starting hierarchical clustering

        Returns: Pandas dataframe with a new column `deduplication_id`. Rows with the same `deduplication_id` are
        deduplicated.

        """
        X[ROW_ID] = np.arange(len(X))
        if self.verbose:
            print('blocking started')
        pairs_table = self.myBlocker.transform(X)
        if self.verbose:
            print('blocking finished')
            print(f'Nr of pairs: {len(pairs_table)}')
            print('scoring started')
        scored_pairs_table = self._calculate_string_similarities(pairs_table)
        scored_pairs_table['score'] = self.myActiveLearner.predict_proba(
            scored_pairs_table['similarities'].tolist())[:, 1]
        scored_pairs_table.loc[
            (scored_pairs_table[[f'{x}_1' for x in self.col_names]].values == scored_pairs_table[
                [f'{x}_2' for x in self.col_names]].values).all(axis=1), 'score'] = 1
        if self.verbose:
            print("scoring finished")
        scored_pairs_table = scored_pairs_table[scored_pairs_table['score'] >= score_threshold]
        if self.verbose:
            print(f'Nr of filtered pairs: {len(scored_pairs_table)}')
            print('Clustering started')
        if self.save_intermediate_steps:
            scored_pairs_table.to_csv('scored_pairs_table.csv', index=None, sep="|")
        df_clusters = hierarchical_clustering(scored_pairs_table, col_names=self.col_names)
        X = X.merge(df_clusters, on=ROW_ID, how='left').drop(columns=[ROW_ID])
        if self.verbose:
            print('Clustering finished')
        X = self._add_singletons(X)
        X[DEDUPLICATION_ID_NAME] = X[DEDUPLICATION_ID_NAME].astype(int)
        return X
