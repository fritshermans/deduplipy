from typing import List, Optional, Union

import numpy as np
import pandas as pd
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

from deduplipy.active_learning.utils_active_learning import input_assert
from deduplipy.classifier_pipeline.classifier_pipeline import ClassifierPipeline
from deduplipy.config import N_QUERIES, MIN_NR_ENTRIES, UNCERTAINTY_IMPROVEMENT_THRESHOLD, UNCERTAINTY_THRESHOLD


class ActiveStringMatchLearner:
    """
    Class to train a string matching model using active learning.

    Args:
        col_names: column names to use for matching
        interaction: whether to include interaction features
        uncertainty_threshold: threshold on the uncertainty of the classifier during active learning,
            used for determining if the model has converged
        uncertainty_improvement_threshold: threshold on the uncertainty *improvement* of classifier during active
            learning, used for determining if the model has converged
        verbose: sets verbosity
        min_nr_entries: minimum number of responses required before classifier convergence is tested

    """

    def __init__(self, col_names: List[str], interaction: bool = False,
                 uncertainty_threshold: float = UNCERTAINTY_THRESHOLD,
                 verbose: Union[int, bool] = 0,
                 uncertainty_improvement_threshold: float = UNCERTAINTY_IMPROVEMENT_THRESHOLD,
                 min_nr_entries: int = MIN_NR_ENTRIES):
        if isinstance(col_names, list):
            self.col_names = col_names
        elif isinstance(col_names, str):
            self.col_names = [col_names]
        else:
            raise Exception('col_name should be list or string')
        self.uncertainty_threshold = uncertainty_threshold
        self.uncertainty_improvement_threshold = uncertainty_improvement_threshold
        self.min_nr_entries = min_nr_entries
        self.verbose = verbose
        self.learner = ActiveLearner(
            estimator=ClassifierPipeline(interaction=interaction),
            query_strategy=uncertainty_sampling,
        )
        self.counter_total = 0
        self.counter_positive = 0
        self.counter_negative = 0

    def _get_last_uncertainty_improvement(self, last_n: int = 5) -> Optional[float]:
        """
        Calculates the uncertainty differences during active learning. The largest difference over the `last_n`
        iterations is returned. The aim of this function is to suggest early stopping of active learning.

        Returns: largest uncertainty update in `last_n` iterations

        """
        uncertainties = np.asarray(self.uncertainties)
        if len(uncertainties) >= last_n + 1:
            differences = abs(uncertainties[1:] - uncertainties[:-1])
            return max(differences[-last_n:])
        else:
            return None

    def _get_active_learning_input(self, query_inst: pd.DataFrame) -> np.ndarray:
        """
        Obtain user input for a query during active learning.

        Args:
            query_inst: query as provided by the ActiveLearner instance

        Returns: label of user input 0 or 1
                    or -1 to go to previous
                    or 9 to finish
                    or 8 to skip the query

        """
        if self.verbose:
            print(f'\nNr. {self.counter_total + 1} ({self.counter_positive}+/{self.counter_negative}-)')
        else:
            print(f'\nNr. {self.counter_total + 1} ({self.counter_positive}+/{self.counter_negative}-)')
        print("Is this a match? (y)es, (n)o, (p)revious, (s)kip, (f)inish")
        with pd.option_context('display.max_colwidth', -1):
            print('->', query_inst[[f'{col_name}_1' for col_name in self.col_names]].iloc[0].to_string())
            print('->', query_inst[[f'{col_name}_2' for col_name in self.col_names]].iloc[0].to_string())
        user_input = input_assert("", ['y', 'n', 'p', 'f', 's'])
        # replace 'y' and 'n' with '1' and '0' to make them valid y labels
        user_input = user_input.replace('y', '1').replace('n', '0')
        # replace 'p' (previous) by '-1', 'f' (finished) by '9', and 's' (skip) by '8'
        user_input = user_input.replace('p', '-1').replace('f', '9').replace('s', '8')
        y_new = np.array([int(user_input)], dtype=int)
        return y_new

    def _print_score_histogram(self, X: pd.DataFrame) -> None:
        """
        Prints histogram of predict_proba scores for X

        Args:
            X: features to calculate predict_proba for

        """
        X_all = pd.concat((self.train_samples, X))
        probas = self.learner.predict_proba(X_all['similarities'].tolist())[:, 1]
        count, division = np.histogram(probas, bins=np.arange(0, 1.01, 0.05))
        hist = pd.DataFrame({'count': count, 'score': division[1:]})
        print(hist[['score', 'count']].to_string(index=False))

    def _print_min_max_scores(self, X):
        X_all = pd.concat((self.train_samples, X))
        pred_max = self.learner.predict_proba(X_all['similarities'].tolist()).max(axis=0)
        print(f'lowest score: {1 - pred_max[0]:.2f}')
        print(f'highest score: {pred_max[1]:.2f}')

    def fit(self, X: pd.DataFrame) -> 'ActiveStringMatchLearner':
        """
        Fit ActiveStringMatchLearner instance on pairs of strings

        Args:
            X: Pandas dataframe containing pairs of strings
        """
        self.uncertainties = []

        self.train_samples = pd.DataFrame([])
        query_inst_prev = None
        uncertainty = None
        for i in range(N_QUERIES):
            query_idx, query_inst = self.learner.query(np.array(X['similarities'].tolist()))
            try:
                uncertainty = 1 - (self.learner.predict_proba(query_inst)[0]).max()
                self.uncertainties.append(uncertainty)
                if self.verbose >= 2:
                    self._print_min_max_scores(X)
            except:
                pass
            y_new = self._get_active_learning_input(X.iloc[query_idx])
            if y_new == -1:  # use previous (input is 'p')
                y_new = self._get_active_learning_input(query_inst_prev)
            elif y_new == 9:  # finish labelling (input is 'f')
                break
            query_inst_prev = X.iloc[query_idx]
            if y_new != 8:  # skip case (input is 's')
                self.learner.teach([X.iloc[query_idx]['similarities'].iloc[0]], y_new)
                train_sample_to_add = X.iloc[query_idx].copy()
                train_sample_to_add['y'] = y_new
                self.train_samples = self.train_samples.append(train_sample_to_add, ignore_index=True)
            X = X.drop(query_idx).reset_index(drop=True)
            last_uncertainty_improvement = self._get_last_uncertainty_improvement()
            if (self.counter_total >= self.min_nr_entries) and last_uncertainty_improvement:
                if self.verbose:
                    print(f"Last uncertainty improvement: {last_uncertainty_improvement:.3f}")
                    print(f'Uncertainty: {uncertainty:.3f}')
                if (uncertainty < self.uncertainty_threshold) | (
                        last_uncertainty_improvement < self.uncertainty_improvement_threshold):
                    print("Classifier converged, enter 'f' to stop training")
            if y_new == 1:
                self.counter_positive += 1
            elif y_new == 0:
                self.counter_negative += 1
            self.counter_total += 1
        if self.verbose:
            self._print_score_histogram(X)
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict on new data whether the pairs are a match or not

        Args:
            X: Pandas dataframe to predict on

        Returns:
            predictions

        """
        return self.learner.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict probabilities on new data whether the pairs are a match or not

        Args:
            X: Pandas dataframe to predict on

        Returns:
            match probabilities

        """
        return self.learner.predict_proba(X)
