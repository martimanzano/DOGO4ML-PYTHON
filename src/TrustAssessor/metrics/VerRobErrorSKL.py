from TrustAssessor.metrics.Metric import Metric
from art.metrics import RobustnessVerificationTreeModelsCliqueMethod
from art.estimators.classification import SklearnClassifier
import pandas as pd

class VerifiedErrorSKLTree(Metric):
    """Verified error of a decision-tree-based model on the provided dataset (TrustableEntity -> data_x, data_y) using the ART package.

    Args:
        Metric (Class): Metric interface
    """
    
    def __init__(self):
        super().__init__()

    def assess(self, trainedModel, dataX, dataY):
        print("TRUST - Computing verified error...")
        rf_skmodel = SklearnClassifier(model=trainedModel)
        rt = RobustnessVerificationTreeModelsCliqueMethod(classifier=rf_skmodel)        
        average_bound, verified_error = rt.verify(x=dataX.values, y=pd.get_dummies(dataY).values, eps_init=0.001,
         nb_search_steps=1, max_clique=2, max_level=1)

        self.assessment = (1-verified_error)

