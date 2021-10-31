import abc


class Sampling:
    def __init__(self, col_names):
        self.col_names = col_names
        self.pairs_col_names = Sampling.get_pairs_col_names(self.col_names)

    @staticmethod
    def get_pairs_col_names(col_names):
        return [f'{x}_1' for x in col_names] + [f'{x}_2' for x in col_names]

    @abc.abstractmethod
    def sample(self, X, n_samples):
        pass
