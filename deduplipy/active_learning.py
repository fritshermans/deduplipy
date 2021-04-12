import numpy as np
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.base import BaseEstimator

from deduplipy.string_matcher import StringMatcher
from deduplipy.utils import input_assert


class ActiveStringMatchLearner(BaseEstimator):
    def __init__(self, X_initial, y_initial, n_queries):
        self.X_initial = X_initial
        self.y_initial = y_initial
        self.n_queries = n_queries
        self.learner = ActiveLearner(
            estimator=StringMatcher(),
            query_strategy=uncertainty_sampling,
            X_training=X_initial, y_training=y_initial
        )

    def _get_lr_params(self):
        return self.learner.estimator.classifier.named_steps['logisticregression'].coef_[0]

    def fit(self, X, y=None):
        X_pool = X.copy()
        self.parameters = [self._get_lr_params()]

        for i in range(self.n_queries):
            query_idx, query_inst = self.learner.query(X_pool)
            print(f'\n{i + 1}/{self.n_queries}', self._get_lr_params())
            print("Is this a match?")
            print('->', query_inst[0][0])
            print('->', query_inst[0][1])
            user_input = input_assert("", ['0', '1', 'y', 'n'])
            user_input = user_input.replace('y', '1').replace('n', '0')
            y_new = np.array([int(user_input)], dtype=int)
            self.learner.teach(query_inst.reshape(1, -1), y_new)
            X_pool = np.delete(X_pool, query_idx, axis=0)
            self.parameters.append(self._get_lr_params())

    def predict(self, X):
        return self.learner.predict(X)

    def predict_proba(self, X):
        return self.learner.predict_proba(X)
