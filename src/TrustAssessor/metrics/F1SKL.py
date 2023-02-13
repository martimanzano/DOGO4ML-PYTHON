from TrustAssessor.metrics.Metric import Metric
from sklearn.metrics import f1_score

class F1SKL(Metric):
    """F1 score for sklearn classifiers, using sklearn. The F1 score can be interpreted 
    as a weighted average of the precision and recall, where an F1 score reaches its best
    value at 1 and worst score at 0. The relative contribution of precision and recall 
    to the F1 score are equal. The formula for the F1 score is:

    F1 = 2 * (precision * recall) / (precision + recall)

    (Extracted from sklearn documentation)

    ADDITIONAL PROPERTIES:
    multiclass_average (str): 'binary' for binary classification problems, for 
    multiclass/multilabel targets, 'micro', 'macro' or 'weighted'.

    Args:
        Metric (Class): Metric interface
    """
    
    def __init__(self, additionalProperties):
        super().__init__()
        self.multiclass_average = additionalProperties["multiclass_average"]
        

    def assess(self, trainedModel, dataX, dataY):
        pred = trainedModel.predict(dataX)
        self.assessment = f1_score(dataY, pred, average=self.multiclass_average)
