from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklego.metrics import p_percent_score
from aix360.metrics import faithfulness_metric, monotonicity_metric
from art.metrics import RobustnessVerificationTreeModelsCliqueMethod
from art.estimators.classification import SklearnClassifier
import numpy as np
import pandas as pd

from Trust_Weighted_Average import Trust_Weighted_Average
from Trust_BN import Trust_BN

class Trust:
    def __init__(self, yaml_config_file, rf_model, data_x, data_y, lime_explainer):
        self.yaml_config_file = yaml_config_file
        self.rf_model = rf_model
        self.data_x = data_x
        self.data_y = data_y
        self.lime_explainer = lime_explainer

        self.feedback = yaml_config_file['general']['feedback']
        self.protected_attributes = yaml_config_file['general']['protected_attributes']
        self.get_performance_metrics_from_dict(self.compute_performance_dict())
        self.fair_p_percentage = self.compute_fairness(self.protected_attributes)
        self.rob_average_bound, self.rob_verified_inv_error = self.compute_robustness()
        self.expl_average_monotonicity, self.expl_average_faithfulness = self.compute_explainability()       

    def compute_performance_dict(self):
        # USE THE INPUT MODEL TO PREDICT THE TEST SET AND REPORT PERFORMANCE METRICS
        print("TRUST - Computing performance metrics...")
        prediction = self.rf_model.predict(self.data_x)
        return classification_report(self.data_y, prediction, output_dict=True)                
    
    def get_performance_metrics_from_dict(self, perf_dict):
        self.perf_accuracy = perf_dict['accuracy']
        self.perf_precision = perf_dict['weighted avg']['precision']
        self.perf_recall = perf_dict['weighted avg']['recall']
        self.perf_f1 = perf_dict['weighted avg']['f1-score']

    def compute_fairness(self, protected_attrs):
        # MODEL P-PERCENTAGE
        print("TRUST - Computing fairness metrics...")
        p_percentage_vector = np.zeros(self.data_x.values.shape[0])
        for i in range(len(protected_attrs)):
            p_percentage_vector[i] = p_percent_score(sensitive_column=protected_attrs[i])(self.rf_model, self.data_x)
        return np.mean(p_percentage_vector)

    def compute_robustness(self):
        # ROBUSTNESS VERIFICATION TREE MODELS CLIQUE METHOD (ART)
        print("TRUST - Computing robustness metrics...")
        rf_skmodel = SklearnClassifier(model=self.rf_model)
        rt = RobustnessVerificationTreeModelsCliqueMethod(classifier=rf_skmodel)
        average_bound, verified_error = rt.verify(x=self.data_x.values, y=pd.get_dummies(self.data_y).values, eps_init=0.3)#, eps_init=0.3, nb_search_steps=10, max_clique=2, 
                                            # max_level=2)
        return average_bound, (1-verified_error) # THE LARGER THE AVRG. BOUND THE BETTER, THE LOWER THE VERIFIED ERROR THE BETTER (SO WE INVERT THE ERROR)

    def compute_explainability(self):
        ### EXPLAINABILITY ###
        print("TRUST - Computing explainability metrics...")
        ncases = self.data_x.values.shape[0]
        monotonicity_vector = np.zeros(ncases) 
        faithfulness_vector = np.zeros(ncases)
        for i in range(ncases):
            print("Computing explainability...Case " + repr(i+1) + "/" + repr(ncases))
            predicted_class = self.rf_model.predict(self.data_x.values[i].reshape(1,-1))[0]
            explanation = self.lime_explainer.explain_instance(self.data_x.values[i], self.rf_model.predict_proba, num_features=5, top_labels=1)
            local_explanation = explanation.local_exp[predicted_class]

            x = self.data_x.values[i]
            coefs = np.zeros(x.shape[0])
        
            for v in local_explanation:
                coefs[v[0]] = v[1]
            base = np.zeros(x.shape[0])

            monotonicity_vector[i] = monotonicity_metric(self.rf_model, self.data_x.values[i], coefs, base)
            faithfulness_vector[i] = faithfulness_metric(self.rf_model, self.data_x.values[i], coefs, base)
        scaler = MinMaxScaler()
        faithfulness_vector_scaled = scaler.fit_transform(faithfulness_vector.reshape(-1,1)) # COMPUTE FROM -1 TO 1, WE SCALED IT TO 0-1 WITH MINMAX    
        return np.mean(monotonicity_vector), np.mean(faithfulness_vector_scaled)

    def compute_trust_weighted_average(self):
        return Trust_Weighted_Average(self)

    def compute_trust_bn(self):
        return Trust_BN(self)

    def get_hash_identifier(self):
        pass