from trust.metrics.metric import Metric

class ExplanationsAccuracyTED(Metric):
    """Accuracy of the TED-enhanced classifier using a test dataset. TED is an explainability framework that leverages domain-relevant explanations in
    the training dataset to predict both labels and explanations for new instances [#]_. (Extracted from TED_Cartesian documentation).

    It requires the TrustableEntity to have a dataset of explanations (Optional parameter data_E in the TrustableEntity initializer).

    Args:
        Metric (Class): Metric interface
    """
    
    def assess(trustable_entity):
        YE_accuracy, Y_accuracy, E_accuracy = trustable_entity.explainer.score(
            trustable_entity.data_x, trustable_entity.data_y, trustable_entity.data_E)
        return YE_accuracy