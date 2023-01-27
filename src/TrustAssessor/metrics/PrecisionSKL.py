from TrustAssessor.metrics.Metric import Metric
from sklearn.metrics import precision_score

class PrecisionSKL(Metric):
    """Precision for sklearn-based classifiers using sklearn. The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.

    The best value is 1 and the worst value is 0.

    (Extracted from sklearn documentation)

    Args:
        Metric (Class): Metric interface
    """
    
    def __init__(self):
        super().__init__()

    def assess(self, trainedModel, dataX, dataY):
        pred = trainedModel.predict(dataX)
        self.assessment = precision_score(dataY, pred)