import numpy as np
from trustML.TrustworthinessComputation.metrics.Metric import Metric
from sklego.metrics import equal_opportunity_score

class EqualOpportunitySKL(Metric):
    """Equal opportunity metric of a sklearn-based classifier using sklego. The equality opportunity 
    score calculates the ratio between the probability of a **true positive** outcome given the sensitive 
    attribute (column) being true and the same probability given the sensitive attribute being false.

    This is especially useful to use in situations where "fairness" is a theme.

    (Extracted from sklego documentation)

    ADDITIONAL PROPERTIES: 
    - protected_attributes (list of str): list of sensible features
    - positive_class (optional): privileged class (if present, 1 otherwise)

    Args:
        Metric (Class): Metric abstract class
    """
    
    def __init__(self, additionalProperties):
        super().__init__()
        self.protectedAttributes = additionalProperties["protected_attributes"]
        if "positive_class" in additionalProperties:
            self.positiveClass = additionalProperties["positive_class"]
        else:
            self.positiveClass = 1

    def assess(self, trainedModel, dataX, dataY):    
        print("Computing equal opportunity metric...")
        if (self.protectedAttributes is None):
            self.value = 1
        else:
            eq_opp_vector = np.zeros(len(self.protectedAttributes))
            for i in range(len(self.protectedAttributes)):
                eq_opp_vector[i] = equal_opportunity_score(sensitive_column = self.protectedAttributes[i], positive_target=self.positiveClass)(trainedModel, dataX, dataY)
                                  
            self.value = np.mean(eq_opp_vector)
