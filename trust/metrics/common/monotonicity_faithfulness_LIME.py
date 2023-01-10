from trust.metrics.metric import Metric
import numpy as np
import tqdm
from sklearn.preprocessing import MinMaxScaler
from aix360.metrics import monotonicity_metric, faithfulness_metric

class MonotonicityFaithfulnessLIME(Metric):
    """Common code used to compute the average monotonicity and average faithfulness metrics. For details on each metric, check the corresponding metric's class.

    Args:
        Metric (Class): Metric interface
    """
    
    def assess(trustable_entity):
        print("TRUST - Computing monotonicity/faithfulness metrics of the model and the LIME explainer...")
        ncases = trustable_entity.data_x.values.shape[0]     
        monotonicity_vector = np.zeros(ncases)
        faithfulness_vector = np.zeros(ncases) 
        for i in tqdm.tqdm(range(ncases), desc="Computing explainability"):
            #predicted_class = rf_model.predict(data_x.values[i].reshape(1,-1))[0]
            explanation = trustable_entity.explainer.explain_instance(
                trustable_entity.data_x.values[i], trustable_entity.trained_model.predict_proba, num_features=5, top_labels=1, num_samples=100)
            local_explanation = explanation.local_exp[next(iter(explanation.local_exp))]#explanation.local_exp[predicted_class]

            x = trustable_entity.data_x.values[i]
            coefs = np.zeros(x.shape[0])
        
            for v in local_explanation:
                coefs[v[0]] = v[1]
            base = np.zeros(x.shape[0])

            monotonicity_vector[i] = monotonicity_metric(trustable_entity.trained_model, trustable_entity.data_x.values[i], coefs, base)
            faithfulness_vector[i] = faithfulness_metric(trustable_entity.trained_model, trustable_entity.data_x.values[i], coefs, base)
        scaler = MinMaxScaler()
        faithfulness_vector_scaled = scaler.fit_transform(faithfulness_vector.reshape(-1,1)) # COMPUTE FROM -1 TO 1, WE SCALED IT TO 0-1 WITH MINMAX

        return np.mean(monotonicity_vector), np.mean(faithfulness_vector_scaled)