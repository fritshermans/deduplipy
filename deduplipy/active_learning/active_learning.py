import pandas as pd
import numpy as np
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

from deduplipy.string_matcher.string_matcher import StringMatcher
from deduplipy.active_learning.utils_active_learning import input_assert


class ActiveStringMatchLearner:
    def __init__(self, n_queries, col_name, coef_diff_threshold=0.05):
        """
        Class to train a string matching model using active learning.

        Args:
            n_queries: number of queries to provide during active learning
            col_name: column name to use for matching
            coef_diff_threshold: threshold on largest update difference in logistic regression parameters, when this
                                threshold is breached, a message is presented that the model had converged
        """
        self.n_queries = n_queries
        self.col_name = col_name
        self.coef_diff_threshold = coef_diff_threshold
        self.learner = ActiveLearner(
            estimator=StringMatcher(self.col_name),
            query_strategy=uncertainty_sampling,
        )

    def _get_lr_params(self):
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

    def _get_largest_coef_diff(self):
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

    def _get_active_learning_input(self, query_inst, learn_counter):
        """
        Obtain user input for a query during active learning.

        Args:
            query_inst: query as provided by the ActiveLearner instance
            learn_counter: integer counting the number of the query

        Returns: label of user input 0 or 1
                    or -1 to go to previous
                    or 9 to finish
                    or 8 to skip the query

        """
        params = self._get_lr_params()
        print(f'\nNr. {learn_counter + 1}', params if isinstance(params, np.ndarray) else '')
        print("Is this a match? (y)es, (n)o, (p)revious, (u)nsure, (f)inish")
        print('->', query_inst[f'{self.col_name}_1'].iloc[0])
        print('->', query_inst[f'{self.col_name}_2'].iloc[0])
        user_input = input_assert("", ['0', '1', 'y', 'n', 'p', 'f', 'u'])
        # replace 'y' and 'n' with '1' and '0' to make them valid y labels
        user_input = user_input.replace('y', '1').replace('n', '0')
        # replace 'p' (previous) by '-1', 'f' (finished) by '9', and 'u' (unsure) by '8'
        user_input = user_input.replace('p', '-1').replace('f', '9').replace('u', '8')
        y_new = np.array([int(user_input)], dtype=int)
        return y_new

    def fit(self, X):
        """
        Fit ActiveStringMatchLearner instance on pairs of strings

        Args:
            X: Pandas dataframe containing pairs of strings
        """
        self.parameters = [self._get_lr_params()]

        query_idx_prev, query_inst_prev = None, None
        learn_counter = 0
        for i in range(self.n_queries):
            query_idx, query_inst = self.learner.query(X)
            y_new = self._get_active_learning_input(query_inst, learn_counter)
            if y_new == -1:  # use previous (input is 'p')
                y_new = self._get_active_learning_input(query_inst_prev, learn_counter)
            elif y_new == 9:  # finish labelling (input is 'f')
                break
            query_idx_prev, query_inst_prev = query_idx, query_inst
            if y_new != 8:  # skip unsure case (input is 'u')
                self.learner.teach(query_inst, y_new)
            X = X.drop(query_idx).reset_index(drop=True)
            self.parameters.append(self._get_lr_params())
            largest_coef_diff = self._get_largest_coef_diff()
            if largest_coef_diff:
                print(largest_coef_diff)
                if largest_coef_diff < self.coef_diff_threshold:
                    print("Classifier converged, enter 'f' to stop training")
            learn_counter += 1

        # print score histogram
        probas = self.learner.predict_proba(X)[:, 1]
        count, division = np.histogram(probas, bins=np.arange(0, 1.01, 0.05))
        hist = pd.DataFrame({'count': count, 'score': division[1:]})
        print(hist)

    def predict(self, X):
        """
        Predict on new data whether the pairs are a match or not

        Args:
            X: Pandas dataframe to predict on

        Returns: predictions

        """
        return self.learner.predict(X)

    def predict_proba(self, X):
        """
        Predict probabilities on new data whether the pairs are a match or not

        Args:
            X: Pandas dataframe to predict on

        Returns: match probabilities

        """
        return self.learner.predict_proba(X)
