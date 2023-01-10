from trust.metrics.metric import Metric
from sklearn.metrics import recall_score

class RecallSKL(Metric):
    """Recall score for sklearn-based classifiers using sklearn.

    The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.

    The best value is 1 and the worst value is 0.

    (Extracted from sklearn documentation)

    Args:
        Metric (Class): Metric interface
    """
    
    def assess(trustable_entity):
        pred = trustable_entity.trained_model.predict(trustable_entity.data_x)
        return recall_score(trustable_entity.data_y, pred)
    
