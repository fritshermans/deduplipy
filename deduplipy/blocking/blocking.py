from typing import List, Union, Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from deduplipy.blocking.set_cover import greedy_set_cover
from deduplipy.config import ROW_ID


class Blocking(BaseEstimator):
    """
    Class for fitting blocking rules and applying them on new pairs

    Args:
        col_names: list of column names, also the ones not included in blocking
        rules_info: dict with column names as keys and a list of blocking functions as values
        recall: minimum recall required
        save_intermediate_steps: whether to save intermediate results

    """
    def __init__(self, col_names: List[str], rules_info: Dict, recall: float = 1.0,
                 save_intermediate_steps: bool = False):
        self.rules_info = rules_info
        self.col_names = col_names
        self._get_rules_specs()
        self.recall = recall
        self.save_intermediate_steps = save_intermediate_steps

    def _get_rules_specs(self):
        self._rules_specs = []
        for col_name, functions in self.rules_info.items():
            for function in functions:
                self._rules_specs.append(
                    {'col_name': col_name, 'function': function, 'function_name': function.__name__})

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

        for j, rule_spec in enumerate(self._rules_specs):
            col_name = rule_spec['col_name']
            function = rule_spec['function']
            col_1 = pd.Series([apply_rule(function, x, j) for x in df_training[f'{col_name}_1']])
            col_2 = pd.Series([apply_rule(function, x, j) for x in df_training[f'{col_name}_2']])
            match_result = (((col_1 != None) & (col_2 != None)) & (col_1 == col_2)).astype(int)
            self._rules_specs[j].update({'rule_set': set((match_result[match_result == 1]).index.tolist())})

        self.subsets = [x['rule_set'] for x in self._rules_specs]

        self.matches = df_training[df_training.match == 1].index.tolist()
        self.universe = set(self.matches)

        self.cover = greedy_set_cover(self.subsets, self.universe, self.recall)

        self.rules_selected = []
        for rule_spec in self._rules_specs:
            if rule_spec['rule_set'] in self.cover:
                self.rules_selected.append(rule_spec)
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
            col_name = rule_selected['col_name']
            func_name = rule_selected['function_name']
            function = rule_selected['function']
            df[f'{col_name}_{func_name}'] = pd.Series([function(x) for x in df[col_name]])
            df.loc[df[f'{col_name}_{func_name}'].notnull(), f'{col_name}_{func_name}'] = \
                df[df[f'{col_name}_{func_name}'].notnull()][f'{col_name}_{func_name}'] + f":{j}"
        df_melted = df.melt(id_vars=self.col_names + [ROW_ID], value_name='fingerprint').drop(columns=['variable'])
        df_melted.dropna(inplace=True)
        return df_melted

    def _create_pairs_table(self, X_fingerprinted: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a pairs table based on the result of fingerprinting

        Args:
            X_fingerprinted: Pandas dataframe containing the fingerprinting result

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
