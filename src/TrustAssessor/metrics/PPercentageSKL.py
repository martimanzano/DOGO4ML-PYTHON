import numpy as np
from TrustAssessor.metrics.Metric import Metric
from sklego.metrics import p_percent_score

class PPercentageSKL(Metric):
    """p_percent metric of a sklearn-based classifier. The p_percent score calculates the ratio between the probability
    of a positive outcome given the sensitive attribute (column) being true and the same probability given the sensitive
    attribute being false.

    This is especially useful to use in situations where "fairness" is a theme.

    (Extracted from sklego documentation)

    The metric is computed for the classification model of the TrustableEntity, along with its data_x and data_y.
    The metric is computed using the protected attributes specified by the user in the configuration YAML/JSON

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
        print("TRUST - Computing p-percentage vector...")
        if (self.protectedAttributes is None):
            self.assessment = 1
        else:
            p_percentage_vector = np.zeros(len(self.protectedAttributes))
            for i in range(len(self.protectedAttributes)):
                p_percentage_vector[i] = p_percent_score(sensitive_column=self.protectedAttributes[i], 
                positive_target=self.positiveClass)(trainedModel, dataX, dataY) 
                                  
            self.assessment = np.mean(p_percentage_vector)
