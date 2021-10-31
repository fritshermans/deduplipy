from itertools import product
from typing import List

import numpy as np
import pandas as pd

from deduplipy.config import ROW_ID, N_PERFECT_MATCHES_TRAIN
from .sampling import Sampling


class NaiveSampling(Sampling):
    def __init__(self, col_names: List[str], n_perfect_matches: int = N_PERFECT_MATCHES_TRAIN):
        super().__init__(col_names)
        self.n_perfect_matches = n_perfect_matches

    def sample(self, X: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        X_pool = X.copy()
        X_pool[ROW_ID] = np.arange(len(X_pool))
        df_sample = X_pool.sample(n=min([len(X_pool), int(n_samples ** 0.5)]))

        pairs_table = pd.DataFrame(
            list(product(df_sample[self.col_names + [ROW_ID]].values.tolist(),
                         df_sample[self.col_names + [ROW_ID]].values.tolist())))

        pairs_table[[f'{x}_1' for x in self.col_names + [ROW_ID]]] = pairs_table[0].to_list()
        pairs_table[[f'{x}_2' for x in self.col_names + [ROW_ID]]] = pairs_table[1].to_list()
        pairs_table.drop(columns=[0, 1], inplace=True)

        pairs_table.sort_values([f'{ROW_ID}_1', f'{ROW_ID}_2'], inplace=True)

        perfect_matches = pairs_table[pairs_table[f'{ROW_ID}_1'] == pairs_table[f'{ROW_ID}_2']].iloc[
                          :self.n_perfect_matches]

        pairs_table = pairs_table[
            pairs_table[f'{ROW_ID}_1'] < pairs_table[f'{ROW_ID}_2']]
        pairs_table = perfect_matches.append(pairs_table, ignore_index=True)
        pairs_table = pairs_table[self.pairs_col_names].reset_index(
            drop=True)
        return pairs_table
