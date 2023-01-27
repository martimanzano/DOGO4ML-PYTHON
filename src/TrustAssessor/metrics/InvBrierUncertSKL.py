from TrustAssessor.metrics.Metric import Metric
from uq360.metrics.classification_metrics import multiclass_brier_score

class InvertedBrierSKL(Metric):
    """Inverted brier score metric of a sklearn classifier using UQ360.    
    
    Args:
        Metric (Class): Metric interface
    """

    def __init__(self):
        super().__init__()

    def assess(self, trainedModel, dataX, dataY):
        print("TRUST - Computing inverted brier uncertainty metric...")
        prediction_proba = trainedModel.predict_proba(dataX)
        
        brier_score = multiclass_brier_score(dataY, prediction_proba)        

        self.assessment = (1-brier_score)