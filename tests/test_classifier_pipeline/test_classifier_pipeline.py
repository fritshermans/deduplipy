import numpy as np

from deduplipy.classifier_pipeline.classifier_pipeline import ClassifierPipeline


def test_base_case():
    myClassifierPipeline = ClassifierPipeline()
    assert list(myClassifierPipeline.classifier.named_steps.keys()) == ['standardscaler', 'logisticregression']
    X = [[0], [1]]
    y = [0, 1]
    myClassifierPipeline.fit(X, y)
    preds = myClassifierPipeline.predict(X)
    pred_proba = myClassifierPipeline.predict_proba(X)
    assert isinstance(preds, np.ndarray)
    np.testing.assert_array_equal(preds, [0, 1])
    assert preds.dtype == np.int64
    assert isinstance(pred_proba, np.ndarray)
    assert pred_proba.shape == (2, 2)
    assert pred_proba.dtype == np.float64
