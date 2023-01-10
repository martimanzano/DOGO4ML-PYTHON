from trust.metrics.metric import Metric
from art.metrics import RobustnessVerificationTreeModelsCliqueMethod
from art.estimators.classification import SklearnClassifier
import pandas as pd

class AverageBoundVerifiedError(Metric):
    """Common code used to compute the average bound and the inverted verified error. For details on each metric, check the corresponding metric's class.

    Args:
        Metric (Class): Metric interface
    """
    
    def assess(trustable_entity):
        print("TRUST - Computing average robustness bound/verified error...")
        rf_skmodel = SklearnClassifier(model=trustable_entity.trained_model)
        rt = RobustnessVerificationTreeModelsCliqueMethod(classifier=rf_skmodel)        
        average_bound, verified_error = rt.verify(x=trustable_entity.data_x.values, y=pd.get_dummies(trustable_entity.data_y).values, eps_init=0.001, 
        nb_search_steps=1, max_clique=2, max_level=1)

        return average_bound, (1-verified_error)