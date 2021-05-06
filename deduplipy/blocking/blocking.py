from typing import List, Optional, Callable, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from deduplipy.blocking.blocking_rules import *
from deduplipy.blocking.set_cover import greedy_set_cover
from deduplipy.config import ROW_ID


class Blocking(BaseEstimator):
    def __init__(self, col_names: Union[List[str], str], rules: Optional[List[Callable]] = None, recall: float = 1.0,
                 save_intermediate_steps: bool = False) -> 'Blocking':
        """
        Class for fitting blocking rules and applying them on new pairs

        Args:
            col_names: column names on which blocking rules should be applied
            rules: list of rules to test for blocking, if None, all available blocking rules are used
            recall: minimum recall required
            save_intermediate_steps: whether or not to save intermediate results

        """
        if isinstance(col_names, list):
            self.col_names = col_names
        elif isinstance(col_names, str):
            self.col_names = [col_names]
        else:
            raise Exception('col_name should be list or string')
        self.rules = rules
        if not self.rules:
            self.rules = all_rules
        self.recall = recall
        self.save_intermediate_steps = save_intermediate_steps

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]) -> 'Blocking':
        """
        Fit Blocking instance on data

        Args:
            X: array containing pairs
            y: array containing whether pairs are a match or not

        Returns:
            fitted instance

        """
        df_training = X.copy()
        df_training['match'] = y

        def apply_rule(rule, x, j):
            result = rule(x)
            if not result:
                return None
            else:
                return result + f":{j}"

        for j, rule in enumerate(self.rules):
            for col_name in self.col_names:
                df_training[f'{col_name}_1_{rule.__name__}'] = df_training.apply(
                    lambda row: apply_rule(rule, row[f'{col_name}_1'], j), axis=1)
                df_training[f'{col_name}_2_{rule.__name__}'] = df_training.apply(
                    lambda row: apply_rule(rule, row[f'{col_name}_2'], j), axis=1)
                df_training[f'{col_name}_{rule.__name__}'] = df_training.apply(
                    lambda row: int(((row[f'{col_name}_1_{rule.__name__}'] != None) &
                                     (row[f'{col_name}_2_{rule.__name__}'] != None)) &
                                    (row[f'{col_name}_1_{rule.__name__}'] == row[f'{col_name}_2_{rule.__name__}'])),
                    axis=1)

        rule_sets = dict()
        for rule in self.rules:
            for col_name in self.col_names:
                rule_sets.update(
                    {f'{col_name}_{rule.__name__}': [rule, rule.__name__, col_name, set(
                        df_training[df_training[f'{col_name}_{rule.__name__}'] == 1][
                            f'{col_name}_{rule.__name__}'].index.tolist())]})
        self.subsets = [x[3] for x in rule_sets.values()]

        self.matches = df_training[df_training.match == 1].index.tolist()
        self.universe = set(self.matches)

        self.cover = greedy_set_cover(self.subsets, self.universe, self.recall)

        self.rules_selected = []
        for rule_name, rule_specs in rule_sets.items():
            rule, rule_name, col_name, rule_set = rule_specs
            if rule_set in self.cover:
                self.rules_selected.append([rule, rule_name, col_name])
        return self

    def _fingerprint(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies blocking rules to data and adds a column 'fingerprint' containing the blocking rules results

        Args:
            X: array containing column to apply blocking rules on

        Returns:
            Pandas dataframe containing a new column 'fingerprint' with the blocking rules results

        """
        df = X.copy()
        for j, rule_selected in enumerate(self.rules_selected):
            rule, rule_name, col_name = rule_selected
            df[f'{col_name}_{rule_name}'] = df[col_name].apply(lambda x: rule(x))
            df.loc[df[f'{col_name}_{rule_name}'].notnull(), f'{col_name}_{rule_name}'] = \
                df[df[f'{col_name}_{rule_name}'].notnull()][f'{col_name}_{rule_name}'] + f":{j}"
        df_melted = df.melt(id_vars=self.col_names + [ROW_ID], value_name='fingerprint').drop(columns=['variable'])
        df_melted.dropna(inplace=True)
        return df_melted

    def _create_pairs_table(self, X_fingerprinted: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a pairs table based on the result of fingerprinting

        Args:
            X_fingerprinted: Pandas dataframe containing the finger printing result

        Returns:
            pairs table
        """
        pairs_table = X_fingerprinted.merge(X_fingerprinted, on='fingerprint', suffixes=('_1', '_2'))
        self.pairs_col_names = [f'{x}_1' for x in self.col_names] + [f'{x}_2' for x in self.col_names]
        pairs_table = pairs_table[pairs_table[f'{ROW_ID}_1'] < pairs_table[f'{ROW_ID}_2']]
        return pairs_table

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies blocking rules on new data

        Args:
            X: Pandas dataframe containing data on which blocking rules should be applied

        Returns:
            Pandas dataframe containing blocking rules applied on new data

        """
        X_fingerprinted = self._fingerprint(X)
        pairs_table = self._create_pairs_table(X_fingerprinted)
        pairs_table = pairs_table.drop_duplicates(subset=[f'{ROW_ID}_1', f'{ROW_ID}_2'])
        if self.save_intermediate_steps:
            pairs_table.to_csv('pairs_table.csv', index=None, sep="|")
        return pairs_table
