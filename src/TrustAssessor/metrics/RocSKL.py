from TrustAssessor.metrics.Metric import Metric
from sklearn.metrics import roc_auc_score

class ROCSKL(Metric):
    """ROC score for sklearn-based classifiers using sklearn. It computes the Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

    (Extracted from sklearn documentation)

    Args:
        Metric (Class): Metric interface
    """
    
    def __init__(self):
        super().__init__()

    def assess(self, trainedModel, dataX, dataY):
        pred = trainedModel.predict(dataX)
        self.assessment = roc_auc_score(dataY, pred)