import numpy as np

import pytest

from deduplipy.classifier_pipeline.classifier_pipeline import ClassifierPipeline


@pytest.mark.parametrize('mode', ['lr', 'nn'])
def test_base_case(mode):
    myClassifierPipeline = ClassifierPipeline(n_features=1, mode=mode)
    X = np.asarray([[0.], [0.1], [0.9], [1.]])
    y = np.asarray([0, 0, 1, 1])
    myClassifierPipeline.fit(X, y)
    preds = myClassifierPipeline.predict(X)
    pred_proba = myClassifierPipeline.predict_proba(X)
    assert isinstance(preds, np.ndarray)
    assert isinstance(pred_proba, np.ndarray)
    assert pred_proba.shape == (4, 2)
