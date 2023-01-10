from trust.metrics.metric import Metric
from sklearn.metrics import roc_auc_score

class ROCSKL(Metric):
    """ROC score for sklearn-based classifiers using sklearn. It computes the Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

    (Extracted from sklearn documentation)

    Args:
        Metric (Class): Metric interface
    """
    
    def assess(trustable_entity):
        pred = trustable_entity.trained_model.predict(trustable_entity.data_x)
        return roc_auc_score(trustable_entity.data_y, pred)
    
