from typing import List, Dict, Optional, Callable, Union

import numpy as np
import pandas as pd

from deduplipy.active_learning.active_learning import ActiveStringMatchLearner
from deduplipy.blocking import Blocking, all_rules
from deduplipy.clustering.clustering import hierarchical_clustering
from deduplipy.config import DEDUPLICATION_ID_NAME, ROW_ID
from deduplipy.sampling.sampler import Sampler
from deduplipy.sampling import MinHashSampler, NaiveSampler
from deduplipy.string_metrics.string_metrics import adjusted_ratio, adjusted_token_sort_ratio


class Deduplicator:
    """
    Deduplicate entries in Pandas dataframe using columns with names `col_names`. Training takes place during a
    short, interactive session (interactive learning).

    Example:
        >>> df = ...
        >>> myDedupliPy = Deduplicator(['name', 'address'])
        >>> myDedupliPy.fit(df)
        >>> myDedupliPy.predict(df)

    The result is a dataframe with a new column `deduplication_id`. Rows with the same `deduplication_id` are
    deduplicated.

    Args:
        col_names: list of column names to be used for deduplication, if `col_names` is provided, `field_info` can
            be set to `None` as it will be neglected
        field_info: dict containing column names as keys and lists of metrics per column name as values, only used
            when col_names is `None`
        interaction: whether to include interaction features
        rules: list of blocking functions to use for all columns or a dict containing column names as keys and lists
            of blocking functions as values, if not provided, all default rules will be used for all columns
        recall: desired recall reached by blocking rules
        save_intermediate_steps: whether to save intermediate results in csv files for analysis
        verbose: sets verbosity

    """
    def __init__(self, col_names: Optional[List[str]] = None, field_info: Optional[Dict] = None,
                 interaction: bool = False, rules: Union[List[Callable], Dict] = None, recall=1.0,
                 save_intermediate_steps: bool = False, verbose: Union[int, bool] = 0):
        if col_names:
            self.col_names = col_names
            self.field_info = {col_name: [adjusted_ratio, adjusted_token_sort_ratio] for col_name in
                               self.col_names}
        else:
            self.field_info = field_info
            self.col_names = list(self.field_info.keys())
        self.interaction = interaction
        self.rules = rules
        if self.rules is None:
            self.rules = all_rules
        if isinstance(self.rules, list):
            self.rules_info = {x: all_rules for x in self.col_names}
        elif isinstance(self.rules, dict):
            self.rules_info = self.rules
        else:
            raise Exception('`rules` must be a list or a dict')
        self.recall = recall
        self.save_intermediate_steps = save_intermediate_steps
        self.verbose = verbose
        self.myActiveLearner = ActiveStringMatchLearner(col_names=self.col_names, interaction=self.interaction,
                                                        verbose=self.verbose)
        self.myBlocker = Blocking(self.col_names, self.rules_info, recall=self.recall,
                                  save_intermediate_steps=self.save_intermediate_steps)
        self.pairs_col_names = Sampler.get_pairs_col_names(self.col_names)

    def __repr__(self):
        repr_dict = {x: self.__dict__[x] for x in
                     ['col_names', 'field_info', 'interaction', 'rules_info', 'recall']}

        field_info_str = dict()
        for key, value in repr_dict['field_info'].items():
            list_str = [x.__name__ for x in value]
            field_info_str.update({key: list_str})
        repr_dict.update({'field_info': field_info_str})

        rules_info_str = dict()
        for key, value in repr_dict['rules_info'].items():
            list_str = [x.__name__ for x in value]
            rules_info_str.update({key: list_str})
        repr_dict.update({'rules_info': rules_info_str})

        repr_str = 'Deduplicator\n'
        for key, value in repr_dict.items():
            repr_str += f'  - {key} = {value}\n'
        return repr_str

    def _create_pairs_table(self, X: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """
        Create sample of pairs

        Args:
            X: Pandas dataframe containing data to be deduplicated
            n_samples: number of sample pairs to be created

        Returns:
            Pandas dataframe containing pairs

        """
        n_samples_minhash = n_samples // 2
        minhash_pairs = MinHashSampler(self.col_names).sample(X, n_samples_minhash)
        # the number of minhash samples can be (much) smaller than n_samples//2, in such case take more random pairs:
        n_samples_naive = n_samples - len(minhash_pairs)
        naive_pairs = NaiveSampler(self.col_names).sample(X, n_samples_naive)
        pairs = naive_pairs.append(minhash_pairs)
        return pairs.drop_duplicates()

    def _calculate_string_similarities(self, X: pd.DataFrame) -> pd.DataFrame:
        X.reset_index(drop=True, inplace=True)  # need to reset the index because of the list comprehension below
        metrics_col_names = []
        for field in self.field_info.keys():
            for metric in self.field_info[field]:
                metrics_col_name = f'{field}_{metric.__name__}'
                X[metrics_col_name] = pd.Series([metric(x, y) for x, y in zip(X[f'{field}_1'], X[f'{field}_2'])])
                metrics_col_names.append(metrics_col_name)
        X['similarities'] = X[metrics_col_names].values.tolist()
        X.drop(columns=metrics_col_names, inplace=True)
        return X

    def fit(self, X: pd.DataFrame, n_samples: int = 10_000) -> 'Deduplicator':
        """
        Fit the deduplicator instance

        Args:
            X: Pandas dataframe to be used for fitting
            n_samples: number of pairs to be created for active learning

        Returns:
            trained deduplicator instance

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
            print([x['col_name'] + " " + x['function_name'] for x in self.myBlocker.rules_selected])
        return self

    @staticmethod
    def _add_singletons(X: pd.DataFrame) -> pd.DataFrame:
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

    def predict(self, X: pd.DataFrame, score_threshold: float = 0.1, cluster_threshold: float = 0.5,
                fill_missing=True) -> pd.DataFrame:
        """
        Predict on new data using the trained deduplicator.

        Args:
            X: Pandas dataframe with column as used when fitting deduplicator instance
            score_threshold: Classification threshold to use for filtering before starting hierarchical clustering
            cluster_threshold: threshold to apply in hierarchical clustering
            fill_missing: whether or not to apply missing value imputation on adjacency matrix

        Returns:
            Pandas dataframe with a new column `deduplication_id`. Rows with the same `deduplication_id` are
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
        df_clusters = hierarchical_clustering(scored_pairs_table, col_names=self.col_names,
                                              cluster_threshold=cluster_threshold, fill_missing=fill_missing)
        X = X.merge(df_clusters, on=ROW_ID, how='left').drop(columns=[ROW_ID])
        if self.verbose:
            print('Clustering finished')
        X = self._add_singletons(X)
        X[DEDUPLICATION_ID_NAME] = X[DEDUPLICATION_ID_NAME].astype(int)
        return X
