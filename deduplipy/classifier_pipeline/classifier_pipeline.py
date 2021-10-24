from typing import Union, List

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l1_l2

from scikeras.wrappers import KerasClassifier


class ClassifierPipeline(BaseEstimator):
    def __init__(self, n_features, mode: str = 'lr', interaction: bool = False):
        """
        Classification pipeline to be used in ActiveStringMatchLearner. Does not throw an error when there is only one
        class in the targets during the first steps in active learning.

        Args:
            n_features: number of features to be used for classifier
            mode: classifier type to use: 'lr' for logistic regression, 'nn' for neural network
            interaction: Whether or not to include interaction features, only applicable for logistic regression

        """
        self.n_features = n_features
        self.mode = mode
        self.interaction = interaction
        if mode == 'lr':
            if interaction:
                self.classifier = make_pipeline(
                    StandardScaler(),
                    PolynomialFeatures(degree=2, interaction_only=True),
                    LogisticRegression(penalty='l1', class_weight='balanced', solver='saga', max_iter=10_000)
                )
            else:
                self.classifier = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(class_weight='balanced')
                )
        elif mode == 'nn':
            self.sk_model = KerasClassifier(ClassifierPipeline._create_model(self.n_features), optimizer='adam',
                                            epochs=100, batch_size=8, verbose=0)
            self.classifier = make_pipeline(StandardScaler(),
                                            self.sk_model)
        else:
            raise Exception('`mode` should be one of `lr` for logistic regression or `nn` for neural network')

    @staticmethod
    def _calculate_layer_sizes(n_features: int) -> List:
        """
        Method to calculate size of layers dependent on the number of features `n_features`.

        Args:
            n_features: number of features

        Returns:
            List of layer sizes

        """
        if n_features == 1:
            layers = [1, 0]
        else:
            layer_1 = n_features
            layer_2 = max([n_features // 2, 2])
            layers = [layer_1, layer_2]
        return layers

    @staticmethod
    def _create_model(n_features):
        """
        Create Keras classification model.

        Args:
            n_features: number of features

        Returns:
            Keras classification model

        """
        layers = ClassifierPipeline._calculate_layer_sizes(n_features)
        model = Sequential()
        model.add(Dense(layers[0], input_shape=(n_features,), activation='relu',
                        kernel_regularizer=l1_l2(l1=0.1, l2=0.1)))
        if layers[1]:
            model.add(Dense(layers[1], activation='relu', kernel_regularizer=l1_l2(l1=0.1, l2=0.1)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', metrics='accuracy')
        return model

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]) -> 'ClassifierPipeline':
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

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict using fitted instance.

        Args:
            X: features

        Returns:
            predictions

        """
        return self.classifier.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict probabilities using fitted instance.

        Args:
            X: features

        Returns:
            predicted probabilities

        """
        return self.classifier.predict_proba(X)
