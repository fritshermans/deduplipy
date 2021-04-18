from itertools import product

import pandas as pd
import numpy as np
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

from deduplipy.string_matcher.string_matcher import StringMatcher
from deduplipy.active_learning.utils_active_learning import input_assert


class ActiveStringMatchLearner:
    def __init__(self, n_queries, col, coef_diff_threshold=0.01):
        self.n_queries = n_queries
        self.col = col
        self.coef_diff_threshold = coef_diff_threshold
        self.learner = ActiveLearner(
            estimator=StringMatcher(),
            query_strategy=uncertainty_sampling,
        )

    def _get_lr_params(self):
        if hasattr(self.learner.estimator.classifier.named_steps['logisticregression'], 'coef_'):
            return self.learner.estimator.classifier.named_steps['logisticregression'].coef_[0]
        else:
            return None

    def _get_largest_coef_diff(self):
        parameters = [x for x in self.parameters if isinstance(x, np.ndarray)]
        if len(parameters) >= 2:
            parameters_np = np.array(parameters)
            diff = np.diff(parameters_np, axis=0)
            return abs(np.diff(parameters_np, axis=0)[-1]).max()
        else:
            return None

    def _get_active_learning_input(self, query_inst, learn_counter):
        params = self._get_lr_params()
        print(f'\nNr. {learn_counter + 1}', params if isinstance(params, np.ndarray) else '')
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
        X_pool['row_number'] = np.arange(len(X_pool))
        df_sample = X_pool.sample(n=int(n_samples ** 0.5))

        sample_combinations = pd.DataFrame(
            list(product(df_sample[[self.col, 'row_number']].values.tolist(),
                         df_sample[[self.col, 'row_number']].values.tolist())),
            columns=[f'{self.col}_1', f'{self.col}_2'])

        for nr in [1, 2]:
            sample_combinations[f'row_number_{nr}'] = sample_combinations[f'{self.col}_{nr}'].str[1]
            sample_combinations[f'{self.col}_{nr}'] = sample_combinations[f'{self.col}_{nr}'].str[0]

        sample_combinations.sort_values(['row_number_1', 'row_number_2'], inplace=True)

        sample_combinations = sample_combinations[
            sample_combinations['row_number_1'] <= sample_combinations['row_number_2']]
        sample_combinations_array = sample_combinations[[f'{self.col}_1', f'{self.col}_2']].values

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
            largest_coef_diff = self._get_largest_coef_diff()
            if largest_coef_diff:
                print(largest_coef_diff)
                if largest_coef_diff < self.coef_diff_threshold:
                    break
            learn_counter += 1

        # print score histogram
        probas = self.learner.predict_proba(sample_combinations_array)[:,1]
        count, division = np.histogram(probas, bins=np.arange(0,1.01,0.05))
        hist = pd.DataFrame({'count':count, 'division':division[1:]})
        print(hist)


    def predict(self, X):
        return self.learner.predict(X)

    def predict_proba(self, X):
        return self.learner.predict_proba(X)
