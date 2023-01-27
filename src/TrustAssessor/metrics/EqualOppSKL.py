import numpy as np
from TrustAssessor.metrics.Metric import Metric
from sklego.metrics import equal_opportunity_score

class EqualOpportunitySKL(Metric):
    """Equal opportunity metric of a sklearn-based classifier. The equality opportunity score calculates the ratio between the probability of a **true positive** outcome given the sensitive attribute (column) being true and the same probability given the sensitive attribute being false.

    This is especially useful to use in situations where "fairness" is a theme.

    (Extracted from sklego documentation)

    The metric is computed for the classification model of the TrustableEntity, along with its data_x and data_y. The metric is computed using the protected attributes specified by the user in the configuration YAML/JSON

    Args:
        Metric (Class): Metric interface
    """
    
    def __init__(self, additionalProperties):
        super().__init__()
        self.protectedAttributes = additionalProperties["protected_attributes"]
        if "positive_class" in additionalProperties:
            self.positiveClass = additionalProperties["positive_class"]
        else:
            self.positiveClass = 1

    def assess(self, trainedModel, dataX, dataY):    
        print("TRUST - Computing equal opportunity metric...")
        if (self.protectedAttributes is None):
            self.assessment = 1
        else:
            eq_opp_vector = np.zeros(len(self.protectedAttributes))
            for i in range(len(self.protectedAttributes)):
                eq_opp_vector[i] = equal_opportunity_score(sensitive_column = self.protectedAttributes[i], positive_target=self.positiveClass)(trainedModel, dataX, dataY)
                                  
            self.assessment = np.mean(eq_opp_vector)
