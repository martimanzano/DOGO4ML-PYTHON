from trust.metrics.metric import Metric
from sklearn.metrics import accuracy_score

class AccuracySKL(Metric):
    """Accuracy classification score for sklearn-based classifiers using sklearn. In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must *exactly* match the corresponding set of labels in y_true (Extracted from sklearn documentation).
    
    Args:
        Metric (Class): Metric interface
    """
        
    def assess(trustable_entity):
        pred = trustable_entity.trained_model.predict(trustable_entity.data_x)
        return accuracy_score(trustable_entity.data_y, pred)
