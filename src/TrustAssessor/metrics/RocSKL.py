from TrustAssessor.metrics.Metric import Metric
from sklearn.metrics import roc_auc_score

class ROCSKL(Metric):
    """ROC score for sklearn-based classifiers using sklearn. It computes the Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

    (Extracted from sklearn documentation)

    ADDITIONAL PROPERTIES:
    multiclass_average (str): 'macro' for binary classification problems, for 
    multiclass/multilabel targets, 'macro' or 'weighted'.

    Args:
        Metric (Class): Metric interface
    """
      
    def __init__(self, additionalProperties):
        super().__init__()
        self.multiclass_average = additionalProperties["multiclass_average"]

    def assess(self, trainedModel, dataX, dataY):
        pred = trainedModel.predict(dataX)
        self.assessment = roc_auc_score(dataY, pred, average=self.multiclass_average)