from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


class ClassifierPipeline(BaseEstimator):
    def __init__(self, interaction=False):
        """
        Classification pipeline to be used in ActiveStringMatchLearner. Does not throw an error when there is only one
        class in the targets during the first steps in active learning.

        Args:
            interaction: Whether or not to include interaction features

        """
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
        """
        Fit the classification pipeline. Does not throw an error when there is only one class in the targets during the
        first steps in active learning.

        Args:
            X: features
            y: target

        Returns:
            fitted instance

        """
        # force the instance not to fit if there is only one class in y, needed for the first steps in active learning
        if len(set(y)) == 1:
            return self
        else:
            self.classifier.fit(X, y)
            return self

    def predict(self, X):
        """
        Predict using fitted instance.

        Args:
            X: features

        Returns:
            predictions

        """
        return self.classifier.predict(X)

    def predict_proba(self, X):
        """
        Predict probabilities using fitted instance.

        Args:
            X: features

        Returns:
            predicted probabilities

        """
        return self.classifier.predict_proba(X)
