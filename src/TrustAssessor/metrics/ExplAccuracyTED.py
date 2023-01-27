from TrustAssessor.metrics.Metric import Metric
import pickle

class ExplanationsAccuracyTED(Metric):
    """Accuracy of the TED-enhanced classifier using a test dataset. TED is an explainability framework that leverages domain-relevant explanations in
    the training dataset to predict both labels and explanations for new instances [#]_. (Extracted from TED_Cartesian documentation).

    It requires the TrustableEntity to have a dataset of explanations (Optional parameter data_E in the TrustableEntity initializer).

    Args:
        Metric (Class): Metric interface
    """
    
    def __init__(self, additionalProperties):
        super().__init__()

        with open(additionalProperties["explainer_path"], 'rb') as explainer_path:
            self.explainer = pickle.load(explainer_path)

        with open(additionalProperties["explanations_path"], 'rb') as explanations_path:
            self.data_E = pickle.load(explanations_path)

    def assess(self, trainedModel, dataX, dataY):
        YE_accuracy, Y_accuracy, E_accuracy = self.explainer.score(
            dataX, dataY, self.data_E)
        self.assessment = YE_accuracy