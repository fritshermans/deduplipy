from itertools import product

import pandas as pd
import numpy as np
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.base import BaseEstimator

from deduplipy.string_matcher.string_matcher import StringMatcher
from deduplipy.active_learning.utils_active_learning import input_assert


class ActiveStringMatchLearner(BaseEstimator):
    def __init__(self, n_queries, col):
        self.n_queries = n_queries
        self.col = col
        self.learner = ActiveLearner(
            estimator=StringMatcher(),
            query_strategy=uncertainty_sampling,
        )

    def _get_lr_params(self):
        if hasattr(self.learner.estimator.classifier.named_steps['logisticregression'], 'coef_'):
            return self.learner.estimator.classifier.named_steps['logisticregression'].coef_[0]
        else:
            return None

    def _get_active_learning_input(self, query_inst, learn_counter):
        print(f'\n{learn_counter + 1}/{self.n_queries}', self._get_lr_params())
        print("Is this a match?")
        print('->', query_inst[0][0])
        print('->', query_inst[0][1])
        user_input = input_assert("", ['0', '1', 'y', 'n', 'p', 'f', 'u'])
        # replace 'y' and 'n' with '1' and '0' to make them valid y labels
        user_input = user_input.replace('y', '1').replace('n', '0')
        # replace 'p' (previous) by '-1', 'f' (finished) by '9', and 'u' (unsure) by '8'
        user_input = user_input.replace('p', '-1').replace('f', '9').replace('u', '8')
        y_new = np.array([int(user_input)], dtype=int)
        return y_new

    def fit(self, X, y=None, n_samples=1_000):
        X_pool = X.copy()
        df_sample = X_pool.sample(n=int(n_samples ** 0.5))
        sample_combinations = pd.DataFrame(
            list(product(df_sample[self.col].values.tolist(), df_sample[self.col].values.tolist())),
            columns=[f'{self.col}_1', f'{self.col}_2'])
        sample_combinations = sample_combinations.drop_duplicates()
        sample_combinations_array = sample_combinations.values

        self.parameters = [self._get_lr_params()]

        query_idx_prev, query_inst_prev = None, None
        learn_counter = 0
        for i in range(self.n_queries):
            query_idx, query_inst = self.learner.query(sample_combinations_array)
            y_new = self._get_active_learning_input(query_inst, learn_counter)
            if y_new == -1:  # use previous (input is 'p')
                y_new = self._get_active_learning_input(query_inst_prev, learn_counter)
            elif y_new == 9:  # finish labelling (input is 'f')
                break
            query_idx_prev, query_inst_prev = query_idx, query_inst
            if y_new != 8:  # skip unsure case (input is 'u')
                self.learner.teach(query_inst.reshape(1, -1), y_new)
            sample_combinations_array = np.delete(sample_combinations_array, query_idx, axis=0)
            self.parameters.append(self._get_lr_params())
            learn_counter += 1

    def predict(self, X):
        return self.learner.predict(X)

    def predict_proba(self, X):
        return self.learner.predict_proba(X)
