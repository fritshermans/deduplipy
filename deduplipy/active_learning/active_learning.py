from typing import List, Optional, Union

import numpy as np
import pandas as pd
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

from deduplipy.active_learning.utils_active_learning import input_assert
from deduplipy.classifier_pipeline.classifier_pipeline import ClassifierPipeline
from deduplipy.config import N_QUERIES


class ActiveStringMatchLearner:
    def __init__(self, col_names: List[str], interaction: bool = False, coef_diff_threshold: float = 0.05,
                 verbose: Union[int, bool] = 0) -> 'ActiveStringMatchLearner':
        """
        Class to train a string matching model using active learning.

        Args:
            col_names: column names to use for matching
            interaction: whether to include interaction features
            coef_diff_threshold: threshold on largest update difference in logistic regression parameters, when this
                                threshold is breached, a message is presented that the model had converged
            verbose: sets verbosity

        """
        if isinstance(col_names, list):
            self.col_names = col_names
        elif isinstance(col_names, str):
            self.col_names = [col_names]
        else:
            raise Exception('col_name should be list or string')
        self.coef_diff_threshold = coef_diff_threshold
        self.verbose = verbose
        self.learner = ActiveLearner(
            estimator=ClassifierPipeline(interaction=interaction),
            query_strategy=uncertainty_sampling,
        )
        self.counter_total = 0
        self.counter_positive = 0
        self.counter_negative = 0

    def _get_lr_params(self) -> Optional[np.ndarray]:
        """
        Returns logistic regression coefficients if the LR model is trained, otherwise `None` is returned

        Returns: Logistic regression parameters

        """
        if hasattr(self.learner.estimator.classifier.named_steps['logisticregression'], 'coef_'):
            intercept = self.learner.estimator.classifier.named_steps['logisticregression'].intercept_[0]
            coefs = self.learner.estimator.classifier.named_steps['logisticregression'].coef_[0]
            params = np.insert(coefs, 0, intercept)
            return params
        else:
            return None

    def _get_largest_coef_diff(self) -> Optional[float]:
        """
        Calculates the differences per logistic regression parameter from between different fits. The largest difference
        across the parameters in the last fit is returned. The aim of this function is to suggest early stopping of
        active learning.

        Returns: largest logistic regression coefficient update in last fit

        """
        parameters = [x for x in self.parameters if isinstance(x, np.ndarray)]
        if len(parameters) >= 2:
            parameters_np = np.array(parameters)
            return abs(np.diff(parameters_np, axis=0)[-1]).max()
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
        params = self._get_lr_params()
        if isinstance(params, np.ndarray):
            params_str = f"\nLR parameters: {params}"
        else:
            params_str = ""
        if self.verbose:
            print(f'\nNr. {self.counter_total + 1} ({self.counter_positive}+/{self.counter_negative}-)', params_str)
        else:
            print(f'\nNr. {self.counter_total + 1} ({self.counter_positive}+/{self.counter_negative}-)')
        print("Is this a match? (y)es, (n)o, (p)revious, (s)kip, (f)inish")
        with pd.option_context('display.max_colwidth', -1):
            print('->', query_inst[[f'{col_name}_1' for col_name in self.col_names]].iloc[0].to_string())
            print('->', query_inst[[f'{col_name}_2' for col_name in self.col_names]].iloc[0].to_string())
        user_input = input_assert("", ['0', '1', 'y', 'n', 'p', 'f', 's'])
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
        probas = self.learner.predict_proba(X['similarities'].tolist())[:, 1]
        count, division = np.histogram(probas, bins=np.arange(0, 1.01, 0.05))
        hist = pd.DataFrame({'count': count, 'score': division[1:]})
        print(hist[['score', 'count']].to_string(index=False))

    def fit(self, X: pd.DataFrame) -> 'ActiveStringMatchLearner':
        """
        Fit ActiveStringMatchLearner instance on pairs of strings

        Args:
            X: Pandas dataframe containing pairs of strings
        """
        self.parameters = [self._get_lr_params()]

        self.train_samples = pd.DataFrame([])
        query_inst_prev = None
        for i in range(N_QUERIES):
            query_idx, query_inst = self.learner.query(np.array(X['similarities'].tolist()))
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
            self.parameters.append(self._get_lr_params())
            largest_coef_diff = self._get_largest_coef_diff()
            if largest_coef_diff:
                if self.verbose:
                    print(f"Largest step in LR coefficients: {largest_coef_diff}")
                if largest_coef_diff < self.coef_diff_threshold:
                    print("Classifier converged, enter 'f' to stop training")
            if y_new == 1:
                self.counter_positive += 1
            elif y_new == 0:
                self.counter_negative += 1
            self.counter_total += 1
        if self.verbose:
            self._print_score_histogram(X)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict on new data whether the pairs are a match or not

        Args:
            X: Pandas dataframe to predict on

        Returns: predictions

        """
        return self.learner.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict probabilities on new data whether the pairs are a match or not

        Args:
            X: Pandas dataframe to predict on

        Returns: match probabilities

        """
        return self.learner.predict_proba(X)
