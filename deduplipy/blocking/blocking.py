import pandas as pd
from sklearn.base import BaseEstimator

from deduplipy.blocking.set_cover import greedy_set_cover
from deduplipy.blocking.blocking_rules import *


class Blocking(BaseEstimator):
    def __init__(self, col_name, rules=None, recall=1.0, cache_tables=False):
        self.col_name = col_name
        self.rules = rules
        if not self.rules:
            self.rules = all_rules
        self.recall = recall
        self.cache_tables = cache_tables

    def fit(self, X, y):
        df_training = pd.DataFrame(X, columns=[f'{self.col_name}_1', f'{self.col_name}_2'])
        df_training['match'] = y

        for j, rule in enumerate(self.rules):
            df_training[f'{self.col_name}_1_{rule.__name__}'] = df_training[f'{self.col_name}_1'].apply(rule) + f":{j}"
            df_training[f'{self.col_name}_2_{rule.__name__}'] = df_training[f'{self.col_name}_2'].apply(rule) + f":{j}"
            df_training[rule.__name__] = df_training.apply(
                lambda row: int(row[f'{self.col_name}_1_{rule.__name__}'] == row[f'{self.col_name}_2_{rule.__name__}']),
                axis=1)

        rule_sets = dict()
        for rule in self.rules:
            rule_sets.update(
                {rule.__name__: (set(df_training[df_training[rule.__name__] == 1][rule.__name__].index.tolist()))})
        self.subsets = rule_sets.values()

        self.matches = df_training[df_training.match == 1].index.tolist()
        self.universe = set(self.matches)

        self.cover = greedy_set_cover(self.subsets, self.universe, self.recall)

        self.rules_selected = []
        for rule_name, rule_set in rule_sets.items():
            if rule_set in self.cover:
                self.rules_selected.append(rule_name)
        return self

    def _fingerprint(self, X):
        df = pd.DataFrame(X, columns=[self.col_name])
        for j, rule in enumerate(self.rules_selected):
            df[rule] = df[self.col_name].apply(lambda x: eval(rule)(x)) + f":{j}"
        df_melted = df.melt(id_vars=self.col_name, value_name='fingerprint').drop(columns=['variable'])
        return df_melted

    def _create_pairs_table(self, X_fingerprinted):
        pairs_table = X_fingerprinted.merge(X_fingerprinted, on='fingerprint', suffixes=('_1', '_2'))
        return pairs_table

    def transform(self, X):
        X_fingerprinted = self._fingerprint(X)
        pairs_table = self._create_pairs_table(X_fingerprinted)
        pairs_table = pairs_table.drop_duplicates(subset=[f'{self.col_name}_1', f'{self.col_name}_2'])
        if self.cache_tables:
            pairs_table.to_csv('pairs_table.csv')
        return pairs_table
