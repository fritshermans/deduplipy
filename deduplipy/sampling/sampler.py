import abc
from typing import List

import pandas as pd


class Sampler(abc.ABC):
    def __init__(self, col_names: List[str]):
        """
        Abstract base class for creating a sample of pairs using columns `col_names`

        Args:
            col_names: list of column names to be used for creating pairs
        """
        self.col_names = col_names
        self.pairs_col_names = Sampler.get_pairs_col_names(self.col_names)

    @staticmethod
    def get_pairs_col_names(col_names: List[str]) -> List[str]:
        """
        Function to create a list of column names that for the resulting pairs table.

        E.g. `col_names` = ['name', 'address']
        will return
            ['name_1', 'address_1', 'name_2', 'address_2']

        Args:
            col_names:

        Returns:
            list of column names for the pairs table
        """
        return [f'{x}_1' for x in col_names] + [f'{x}_2' for x in col_names]

    @abc.abstractmethod
    def sample(self, X: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        pass
