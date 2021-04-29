from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class ClassifierPipeline(BaseEstimator):
    def __init__(self):
        self.classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight='balanced')
        )
        # force the instance to be fitted such that one can predict during active learning before the model is fitted
        self._fitted = True

    def fit(self, X, y):
        # force the instance not to fit if there is only one class in y, needed for the first steps in active learning
        if len(set(y)) == 1:
            return self
        else:
            self.classifier.fit(X, y)
            return self

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def score(self, X, y):
        preds = self.predict(X)
        return f1_score(preds, y)
