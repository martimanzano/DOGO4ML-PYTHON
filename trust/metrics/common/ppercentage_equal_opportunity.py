import numpy as np
from trust.metrics.metric import Metric
from sklego.metrics import p_percent_score, equal_opportunity_score

class PPercentageEqualOppportunity(Metric):
    """Common code used to compute the average p-percent score and the average equal opportunity score. For details on each metric, check the corresponding metric's class.

    Args:
        Metric (Class): Metric interface
    """
    
    def assess(trustable_entity):
        print("TRUST - Computing p-percentage/equal opportunity...")
        if (trustable_entity.protected_attributes is None):
            return 1
        else:
            p_percentage_vector = np.zeros(len(trustable_entity.protected_attributes))#np.zeros(data_x.values.shape[0])
            eq_opp_vector = np.zeros(len(trustable_entity.protected_attributes))
            for i in range(len(trustable_entity.protected_attributes)):
                p_percentage_vector[i] = p_percent_score(sensitive_column=trustable_entity.protected_attributes[i])(trustable_entity.trained_model, trustable_entity.data_x)
                eq_opp_vector[i] = equal_opportunity_score(trustable_entity.protected_attributes[i])(trustable_entity.trained_model, trustable_entity.data_x, trustable_entity.data_y)
                                  
            return np.mean(p_percentage_vector), np.mean(eq_opp_vector)