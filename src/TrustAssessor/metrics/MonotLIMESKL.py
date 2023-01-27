from TrustAssessor.metrics.Metric import Metric
import pickle
import numpy as np
from aix360.metrics import monotonicity_metric

class MonotonicityLIMESKL(Metric):
    """Average Monotonicity metric of a sklearn classifier and a LIME explainer using the AIX360 package.

    This metric measures the effect of individual features on model performance by evaluating the effect on model performance of incrementally adding each attribute in order of increasing importance. As each feature is added, the performance of the model should correspondingly increase, thereby resulting in monotonically increasing model performance. [#]_

    (Extracted from AIX360 documentation)

    It requires the TrustableEntity to have a LIME tabular explainer (Optional parameter explainer in the TrustableEntity initializer).

    Args:
        Metric (Class): Metric interface
    """
    
    def __init__(self, additionalProperties):
        super().__init__()

        with open(additionalProperties["explainer_path"], 'rb') as explainer_path:
            self.explainer = pickle.load(explainer_path)

    def assess(self, trainedModel, dataX, dataY):
        print("TRUST - Computing monotonicity metric with LIME...")
        ncases = dataX.values.shape[0]     
        monotonicity_vector = np.zeros(ncases) 
        for i in range(ncases):
            print("Case " + str(i+1) + "/" + str(ncases))
            explanation = self.explainer.explain_instance(
                dataX.values[i], trainedModel.predict_proba, num_features=5, top_labels=1, num_samples=100)
            local_explanation = explanation.local_exp[next(iter(explanation.local_exp))]#explanation.local_exp[predicted_class]

            x = dataX.values[i]
            coefs = np.zeros(x.shape[0])
        
            for v in local_explanation:
                coefs[v[0]] = v[1]
            base = np.zeros(x.shape[0])

            monotonicity_vector[i] = monotonicity_metric(trainedModel, dataX.values[i], coefs, base)

        self.assessment = np.mean(monotonicity_vector)
    
