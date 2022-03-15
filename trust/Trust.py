from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from aix360.algorithms.ted.TED_Cartesian import TED_CartesianExplainer
from sklego.metrics import p_percent_score, equal_opportunity_score
from aix360.metrics import faithfulness_metric, monotonicity_metric
from uq360.metrics.classification_metrics import multiclass_brier_score, expected_calibration_error
from art.metrics import RobustnessVerificationTreeModelsCliqueMethod
from art.estimators.classification import SklearnClassifier
import numpy as np
import pandas as pd
import yaml
import pickle
import json
import tqdm
from pathlib import Path
from enum import Enum
from trust.Trust_Weighted_Average import Trust_Weighted_Average
from trust.Trust_BN import Trust_BN

class Explainability_method(Enum):
    """Class used to control which explainability method is used in every Trust instance. For internal use.
    Args:
        Enum (Enum): Explainability method to use
    """
    LIME = 1
    TED = 2

class Evaluation_method(Enum):
    """Class used to specify and control which assessment method is used to compute the Trust in every Trust instance.

    Args:
        Enum (Enum): Assessment method to use
    """
    Weighted_Average = 1
    Bayesian_Network = 2

class Trust:
    """Class that computes and optionally stores the TRUST metrics in a cache file for a sklearn classifier and a specified dataset. 
    It uses some Trust-based libraries such as AIX360, UQ360, and ART360.
    """

    def __init__(self, yaml_config_file, data_x, data_y, trained_model, lime_explainer = None, data_E = None):
        """Entry point of the Trust library without cache management. Instanciates and initializes a Trust object with unweighted Trust metrics computed.
        Depending on the explainability method to use (i.e., LIME or TED), trained_model will be a TED-enhanced classifier or simply a sklearn classifier.
        Lime_explainer is required when using LIME as explainability method, whereas data_E (dataset true explanations) is required when using TED.

        Args:
            yaml_config_file ([yaml module object]): [yaml dict containing the user parameters]
            data_x ([array]): [feature array]
            data_y ([array]): [true labels array]
            trained_model ([sklearn classifier]): [Trained classification model]
            lime_explainer ([LimeTabularExplainer], optional): [LIME explainer previously created]
            data_E ([array], optional): [true explanations array]
        """
        self.feedback = self.get_yaml_feedback(yaml_config_file)
        self.protected_attributes = yaml_config_file['protected_attributes']
        if (isinstance(trained_model, TED_CartesianExplainer)):
            rf_model = trained_model.model
            self.explainability_method = Explainability_method.TED
            self.expl_E_accuracy = self.compute_explainability_TED(trained_model, data_x, data_y, data_E)
            self.expl_average_monotonicity = self.expl_average_faithfulness = None
        elif (isinstance(trained_model, RandomForestClassifier)):
            rf_model = trained_model
            self.explainability_method = Explainability_method.LIME
            self.expl_average_monotonicity, self.expl_average_faithfulness = self.compute_explainability_LIME(rf_model, data_x, data_y, lime_explainer)
            self.expl_E_accuracy = None

        self.perf_accuracy, self.perf_precision, self.perf_recall, self.perf_f1 = self.get_performance_metrics_from_dict(self.compute_performance_dict(rf_model, data_x, data_y))
        self.fair_p_percentage, self.fair_equal_opportunity_score = self.compute_fairness(rf_model, data_x, data_y)
        self.rob_average_bound, self.rob_verified_inv_error = self.compute_robustness(rf_model, data_x, data_y)                
        self.unc_brier_inv_score, self.unc_expected_cal_inv_error = self.compute_uncertainty(rf_model, data_x, data_y)
       
    def get_yaml_feedback(self, yaml_config_file):
        """TRUST: PERFORMANCE. Function to retrieve and return the feedback unweighted value from the yaml configuration file.

        Args:
            yaml_config_file ([yaml module object]): [yaml dict containing the user parameters]

        Returns:
            [float]: [unweighted feedback value]
        """

        return yaml_config_file['feedback']['value']

    def compute_performance_dict(self, rf_model, data_x, data_y):
        """TRUST: PERFORMANCE. Function to compute the performance metrics over a provided dataset and using a provided classification model.

        Args:
            rf_model ([sklearn classifier]): [Trained classification model]
            data_x ([array]): [feature array]
            data_y ([array]): [true labels array]

        Returns:
            [dict]: [dictionary containing the performance metrics evaluated comparing the provided true labels to the predicted ones]
        """

        print("TRUST - Computing performance metrics...")
        prediction = rf_model.predict(data_x)

        return classification_report(data_y, prediction, output_dict=True)                
    
    def get_performance_metrics_from_dict(self, perf_dict):
        """TRUST: PERFORMANCE. Retrieves the main performance metrics from the performance dictionary using sklearn.

        Args:
            perf_dict ([dict]): [dictionary containing the performance metrics, returned using the classification_report function from sklearn]

        Returns:
            [float, float, float, float]: [accuracy, precision, recall and f1 metrics]
        """

        perf_accuracy = perf_dict['accuracy']
        perf_precision = perf_dict['weighted avg']['precision']
        perf_recall = perf_dict['weighted avg']['recall']
        perf_f1 = perf_dict['weighted avg']['f1-score']

        return perf_accuracy, perf_precision, perf_recall, perf_f1

    def compute_fairness(self, rf_model, data_x, data_y):
        """TRUST: FAIRNESS. Function that computes two fairness metrics using the sklego library, i.e., P-Percentage and Equal Opportunity.
        It uses the protected attributes specified in the yaml configuration file. If there are no protected attributes,
        the fairness metrics are returned as 1, as there are no fairness issues.

        Args:
            rf_model ([sklearn classifier]): [Trained classification model]
            data_x ([array]): [feature array]
            data_y ([array]): [true labels array]

        Returns:
            [float, float]: [Means of the p-percentage and equal opportunity vectors, computed over the set of protected attributes]
        """

        print("TRUST - Computing fairness metrics...")
        if (self.protected_attributes is None):
            return 1, 1
        else:
            p_percentage_vector = np.zeros(len(self.protected_attributes))#np.zeros(data_x.values.shape[0])
            eq_opp_vector = np.zeros(len(self.protected_attributes))
            for i in range(len(self.protected_attributes)):
                p_percentage_vector[i] = p_percent_score(sensitive_column=self.protected_attributes[i])(rf_model, data_x)
                eq_opp_vector[i] = equal_opportunity_score(self.protected_attributes[i])(rf_model, data_x, data_y)
                                  
            return np.mean(p_percentage_vector), np.mean(eq_opp_vector)

    def compute_robustness(self, rf_model, data_x, data_y):
        """TRUST: ROBUSTNESS. Function that computes robustness metrics of a sklearn tree-based classifier over a dataset.
        Currently, it uses the Clique method RobustnessVerificationTreeModelsCliqueMethod from the ART360 library.

        Args:
            rf_model ([sklearn classifier]): [Trained classification model. For now, it has to be a sklearn tree-based classifier]
            data_x ([array]): [feature array]
            data_y ([array]): [true labels array]

        Returns:
            [float, float]: [Robustness metrics: average bound, and the inverse of the verified error]
        """
        print("TRUST - Computing robustness metrics...")
        rf_skmodel = SklearnClassifier(model=rf_model)
        rt = RobustnessVerificationTreeModelsCliqueMethod(classifier=rf_skmodel)        
        average_bound, verified_error = rt.verify(x=data_x.values, y=pd.get_dummies(data_y).values, eps_init=0.001,
         nb_search_steps=1, max_clique=2, max_level=1)

        return average_bound, (1-verified_error) # THE LARGER THE AVRG. BOUND THE BETTER, THE LOWER THE VERIFIED ERROR THE BETTER (SO WE INVERT THE ERROR)

    def compute_explainability_LIME(self, rf_model, data_x, data_y, lime_explainer):
        """TRUST: EXPLAINABILITY. Function that computes AIX360 explainability metrics used a previously created LIME explainer over a provided dataset.

        Args:
            rf_model ([sklearn classifier]): [Trained classification model]
            data_x ([array]): [feature array]
            data_y ([array]): [true labels array]
            lime_explainer ([LimeTabularExplainer]): [LIME explainer previously created]

        Returns:
            [float, float]: [Means of the monotonicity and faithfulness vectors, computed over a provided dataset using the provided explainer]
        """

        print("TRUST - Computing explainability metrics...")
        ncases = data_x.values.shape[0]     
        monotonicity_vector = np.zeros(ncases) 
        faithfulness_vector = np.zeros(ncases)
        for i in tqdm.tqdm(range(ncases), desc="Computing explainability"):
            #print("Computing explainability...Case " + repr(i+1) + "/" + repr(ncases))
            #predicted_class = rf_model.predict(data_x.values[i].reshape(1,-1))[0]
            explanation = lime_explainer.explain_instance(data_x.values[i], rf_model.predict_proba, num_features=5, top_labels=1, num_samples=100)
            local_explanation = explanation.local_exp[next(iter(explanation.local_exp))]#explanation.local_exp[predicted_class]

            x = data_x.values[i]
            coefs = np.zeros(x.shape[0])
        
            for v in local_explanation:
                coefs[v[0]] = v[1]
            base = np.zeros(x.shape[0])

            monotonicity_vector[i] = monotonicity_metric(rf_model, data_x.values[i], coefs, base)
            faithfulness_vector[i] = faithfulness_metric(rf_model, data_x.values[i], coefs, base)
        scaler = MinMaxScaler()
        faithfulness_vector_scaled = scaler.fit_transform(faithfulness_vector.reshape(-1,1)) # COMPUTE FROM -1 TO 1, WE SCALED IT TO 0-1 WITH MINMAX

        return np.mean(monotonicity_vector), np.mean(faithfulness_vector_scaled)

    def compute_explainability_TED(self, ted_model, data_x, data_y, data_E):
        """TRUST: EXPLAINABILITY. Function that computes explainability metrics used a previously created TED-enhanced explainer over a provided dataset.
        In this case, the TED framework works with dataset enhanced with explanation.

        Args:
            ted_model ([TED_CartesianExplainer]): [TED-enhanced explainer]
            data_x ([array]): [feature array]
            data_y ([array]): [true labels array]
            data_E ([array]): [true explanations array]

        Returns:
            [float]: [accuracy of the TED-enhanced classifier predicting the explanations]
        """

        YE_accuracy, Y_accuracy, E_accuracy = ted_model.score(data_x, data_y, data_E)    # evaluate the classifier, although we only need E_acc

        return YE_accuracy

    def compute_uncertainty(self, rf_model, data_x, data_y):
        """TRUST: UNCERTAINTY. Computes UQ360 uncertainty metrics over a provided dataset and using a provided classification model.

        Args:
            rf_model ([sklearn classifier]): [Trained classification model]
            data_x ([array]): [feature array]
            data_y ([array]): [true labels array]

        Returns:
            [float, float]: [Inverted brier score and inverted expected calibration error]
        """
        
        print("TRUST - Computing uncertainty metrics...")
        prediction = rf_model.predict(data_x)
        prediction_proba = rf_model.predict_proba(data_x)
        
        brier_score = multiclass_brier_score(data_y, prediction_proba)        
        expected_cal_error = expected_calibration_error(data_y, prediction_proba, prediction, len(set(data_y)), False)

        return (1-brier_score), (1-expected_cal_error) # A "cost function" and an error, therefore we invert it
        
    def evaluate_trust(self, yaml_config_file, computation_method):
        """Function that assesses the trust using one of the provided evaluation methods once the trust metrics have been computed.

        Args:
            yaml_config_file ([yaml module object]): [yaml dict containing the user parameters]
            computation_method ([Evaluation_method]): [chosen assessment method (see class Evaluation_method)]

        Returns:
            [Trust_Weighted_Average or Trust_BN]: [Assessed Trust using the chosen assessment evaluation method]
        """
        with open(yaml_config_file, mode='r') as config_file:
            yaml_config = yaml.safe_load(config_file)
            print("INFO: Using configuration file " + yaml_config_file)
            if (computation_method is Evaluation_method.Weighted_Average):
                return Trust_Weighted_Average(self, yaml_config)
            elif (computation_method is Evaluation_method.Bayesian_Network):
                return Trust_BN(self, yaml_config)

    def compute_trust_bn(self, yaml_config_file):
        """Function that instances and initializes an object of the Trust_BN class to assess the trust using a Bayesian network.

        Args:
            yaml_config_file ([yaml module object]): [yaml dict containing the user parameters, including the Bayesian network filepath and parameters]

        Returns:
            [Trust_BN]: [Trust_BN object initialized to enable the trust assessment using a provided Bayesian network]
        """
        with open(yaml_config_file, mode='r') as config_file:
            yaml_config = yaml.safe_load(config_file)
            print("INFO: Using configuration file " + yaml_config_file)
            return Trust_BN(self, yaml_config)
    
    @staticmethod
    def load_compute_trust_with_cache(file_dataset_name, file_configuration_to_use, data_x, data_y, trained_model, lime_explainer = None, data_E = None):
        """Entry point of the Trust library with cache management. Uses pickle as cache instrument in order to save and load already computed Trust objects.
        Depending on the explainability method to use (i.e., LIME or TED), trained_model will be a TED-enhanced classifier or simply a sklearn classifier.
        Lime_explainer is required when using LIME as explainability method, whereas data_E (dataset true explanations) is required when using TED.

        Args:
            file_dataset_name ([string]): [name of the dataset in order to save/load the appropiate cache file]
            file_configuration_to_use ([string]): [path to the yaml configuration file containing the user parameters]
            data_x ([array]): [feature array]
            data_y ([array]): [true labels array]
            trained_model ([sklearn classifier]): [Trained classification model]
            lime_explainer ([LimeTabularExplainer], optional): [LIME explainer previously created]
            data_E ([array], optional): [true explanations array]
        Returns:
            [Trust]: [Instanced and initialized Trust object with unweighted metrics computed]
        """
        trust = None
        with open(file_configuration_to_use, mode='r') as config_file:
            yaml_config = yaml.safe_load(config_file)
            print("INFO: Using configuration file " + file_configuration_to_use)

            try:
                cached_trust_file_path = Path('cache/trust_' + file_dataset_name + '.cache')
                if cached_trust_file_path.is_file():
                    with open(cached_trust_file_path, 'rb') as cached_trust_file:
                        trust = pickle.load(cached_trust_file)
                        print("INFO: Loaded trust from cache file " + str(cached_trust_file_path.absolute()))
            except:
                pass
            if trust == None:
            # LOAD YAML CONFIG FILE
                print("INFO: Trust cache not found in " + str(cached_trust_file_path.absolute()))
                print("Computing Trust...")
                if (isinstance(trained_model, TED_CartesianExplainer)):
                    trust = Trust(yaml_config, data_x, data_y, trained_model, lime_explainer=None, data_E=data_E)
                else:
                    trust = Trust(yaml_config, data_x, data_y, trained_model, lime_explainer=lime_explainer)               
                                
                with open(cached_trust_file_path, 'wb') as cached_trust_file:
                    pickle.dump(trust, cached_trust_file)

        return trust

    def get_Trust_metrics_as_JSON(self):
        """Returns the unweighted, computed trust metrics as a JSON"""
        # Create Dictionary
        trust_metrics = {
            "ABOUT": "TRUST ASSESSED AND UNWEIGHTED METRICS",
            "feedback": self.feedback,                
            "performance":
                {
                    "accuracy": self.perf_accuracy,                                       
                    "precision": self.perf_precision,                        
                    "recall": self.perf_recall                        
                },
            "fairness":
                {                    
                    "p_percentage": self.fair_p_percentage,                        
                    "equal_opportunity": self.fair_equal_opportunity_score
                        
                },
            "robustness":
                {                    
                    "average_bound": self.rob_average_bound,                        
                    "verified_error_inv": self.rob_verified_inv_error                        
                },
            "explainability":
                {                    
                    "average_monotonicity_LIME": self.expl_average_monotonicity,                        
                    "average_faithfulness_LIME": self.expl_average_faithfulness,                        
                    "E_score_TED": self.expl_E_accuracy                              
                },
            "uncertainty":
                {                    
                    "brier_score": self.unc_brier_inv_score,
                    "expected_calibration_error_inv": self.unc_expected_cal_inv_error                    
                }
        }
        return json.dumps(trust_metrics, indent=4)