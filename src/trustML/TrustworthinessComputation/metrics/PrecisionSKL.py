from trustML.TrustworthinessComputation.metrics.Metric import Metric
from sklearn.metrics import precision_score

class PrecisionSKL(Metric):
    """Precision for sklearn-based classifiers using sklearn. The precision is the ratio tp / (tp + fp)
    where tp is the number of true positives and fp the number of false positives. The precision is 
    intuitively the ability of the classifier not to label as positive a sample that is negative.

    The best value is 1 and the worst value is 0.

    (Extracted from sklearn documentation)

    ADDITIONAL PROPERTIES:
    multiclass_average (str): 'binary' for binary classification problems, for 
    multiclass/multilabel targets, 'micro', 'macro' or 'weighted'.

    Args:
        Metric (Class): Metric abstract class
    """
    
    def __init__(self, additionalProperties):
        super().__init__()
        self.multiclass_average = additionalProperties["multiclass_average"]

    def assess(self, trainedModel, dataX, dataY):
        pred = trainedModel.predict(dataX)
        self.value = precision_score(dataY, pred, average=self.multiclass_average)