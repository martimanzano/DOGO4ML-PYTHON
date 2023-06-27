from trustML.TrustworthinessComputation.metrics.Metric import Metric
from sklearn.metrics import recall_score

class RecallSKL(Metric):
    """Recall score for sklearn-based classifiers using sklearn. The recall is the ratio tp / (tp + fn) 
    where tp is the number of true positives and fn the number of false negatives. 
    The recall is intuitively the ability of the classifier to find all the positive samples.

    The best value is 1 and the worst value is 0.

    (Extracted from sklearn documentation)

    ADDITIONAL PROPERTIES:
    multiclass_average (str): 'binary' for binary classification problems, for 
    multiclass/multilabel targets, 'micro', 'macro', 'samples' or 'weighted'.

    Args:
        Metric (Class): Metric abstract class
    """
    
    def __init__(self, additionalProperties):
        super().__init__()
        self.multiclass_average = additionalProperties["multiclass_average"]

    def assess(self, trainedModel, dataX, dataY):
        pred = trainedModel.predict(dataX)
        self.value = recall_score(dataY, pred, average=self.multiclass_average)