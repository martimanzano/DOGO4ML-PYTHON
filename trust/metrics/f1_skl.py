from trust.metrics.metric import Metric
from sklearn.metrics import f1_score

class F1SKL(Metric):
    """F1 score for sklearn-based classifiers, using sklearn. The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:

    F1 = 2 * (precision * recall) / (precision + recall)

    (Extracted from sklearn documentation)

    Args:
        Metric (Class): Metric interface
    """
    
    def assess(trustable_entity):
        pred = trustable_entity.trained_model.predict(trustable_entity.data_x)
        return f1_score(trustable_entity.data_y, pred)
    
