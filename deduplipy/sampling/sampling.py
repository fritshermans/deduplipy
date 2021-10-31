import abc
from typing import List

import pandas as pd


class Sampling(abc.ABC):
    def __init__(self, col_names: List[str]):
        self.col_names = col_names
        self.pairs_col_names = Sampling.get_pairs_col_names(self.col_names)

    @staticmethod
    def get_pairs_col_names(col_names: List[str]) -> List[str]:
        return [f'{x}_1' for x in col_names] + [f'{x}_2' for x in col_names]

    @abc.abstractmethod
    def sample(self, X: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        pass
