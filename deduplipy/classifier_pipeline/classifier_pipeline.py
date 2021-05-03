from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


class ClassifierPipeline(BaseEstimator):
    def __init__(self, interaction=False):
        if interaction:
            self.classifier = make_pipeline(
                StandardScaler(),
                PolynomialFeatures(degree=2, interaction_only=True),
                LogisticRegression(penalty='l1', class_weight='balanced', solver='saga', max_iter=1_000)
            )
        else:
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
