import numpy as np
from fuzzywuzzy.fuzz import ratio, token_set_ratio, token_sort_ratio, partial_ratio
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class StringMatcher(BaseEstimator):
    def __init__(self):
        self.classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight='balanced')
        )
        # force the instance to be fitted such that one can predict during active learning before the model is fitted
        self._fitted = True

    def _similarities(self, X):
        return np.asarray([[ratio(x[0], x[1]),
                            partial_ratio(x[0], x[1]),
                            token_set_ratio(x[0], x[1]),
                            token_sort_ratio(x[0], x[1])] for x in X])

    def fit(self, X, y):
        # force the instance not to fit if there is only one class in y, needed for the first steps in active learning
        if len(set(y))==1:
            return self
        else:
            similarities = self._similarities(X)
            self.classifier.fit(similarities, y)
            return self

    def predict(self, X):
        similarities = self._similarities(X)
        return self.classifier.predict(similarities)

    def predict_proba(self, X):
        similarities = self._similarities(X)
        return self.classifier.predict_proba(similarities)

    def score(self, X, y):
        preds = self.predict(X)
        return f1_score(preds, y)
