import trust.Trust as Trust
import json

class Trust_Weighted_Average():
    """Class that evaluates the Trust as a weighted average, by using the weights specified in the yaml configuration file.
    It requires an instance of a Trust object, which contains the trust metrics already computed and unweighted.
    """
    
    def __init__(self, trust_instance : Trust, yaml_config_file):
        """Initializer. Retrieves the weights from the yaml configuration file and assesses the trust with its metrics and
        aggregations, using the Trust instance passed as parameter.

        Args:
            trust_instance (Trust): [Instanced and initialized Trust object with unweighted metrics computed]
            yaml_config_file ([yaml module object]): [yaml dict containing the user parameters]

        Raises:
            ValueError: [When any set of related weights do not add 1]
        """
        self.trust_instance = trust_instance
        self.trust_instance.feedback = self.trust_instance.get_yaml_feedback(yaml_config_file)
        
        self.weight_feedback = yaml_config_file['feedback']['weight']
        self.weight_performance = yaml_config_file['w.avg_parameters']['performance']['weight']
        self.weight_fairness = yaml_config_file['w.avg_parameters']['fairness']['weight']
        self.weight_robustness = yaml_config_file['w.avg_parameters']['robustness']['weight']
        self.weight_explainability = yaml_config_file['w.avg_parameters']['explainability']['weight']
        self.weight_uncertainty = yaml_config_file['w.avg_parameters']['uncertainty']['weight']

        if round(self.weight_performance + self.weight_fairness + self.weight_robustness + self.weight_explainability
         + self.weight_uncertainty + self.weight_feedback, 2) != 1:
            raise ValueError('Factor weights do not add 1 (' + 
            repr(self.weight_performance + self.weight_fairness + self.weight_robustness + self.weight_explainability 
            + self.weight_uncertainty + self.weight_feedback) + '). Revise the configuration file.')

        # FEEDBACK #
        self.score_feedback = self.compute_score_feedback()
        # PERFORMANCE #
        self.weight_perf_accuracy, self.weight_perf_precision, self.weight_perf_recall, self.weight_perf_f1 = self.load_weights_performance(yaml_config_file)
        self.score_perf_accuracy, self.score_perf_precision, self.score_perf_recall, self.score_perf_f1, self.score_performance = self.compute_score_performance()
        # FAIRNESS #
        self.weight_fair_p_percentage, self.weight_fair_equal_opportunity_score = self.load_weights_fairness(yaml_config_file)
        self.score_fair_p_percentage, self.score_fair_equal_opportunity, self.score_fairness = self.compute_score_fairness()
        # ROBUSTNESS #
        self.weight_rob_average_bound, self.weight_rob_verified_inv_error = self.load_weights_robustness(yaml_config_file)
        self.score_rob_average_bound, self.score_rob_verified_inv_error, self.score_robustness = self.compute_score_robustness()
        # EXPLAINABILITY #
        self.score_explainability = self.weight_expl_average_monotonicity = self.weight_expl_average_faithfulness = self.weight_expl_E_accuracy_score = None
        self.score_expl_average_monotonicity = self.score_expl_average_faithfulness = self.score_expl_E_accuracy_score = None
        self.load_weights_score_explainability(yaml_config_file, self.trust_instance.explainability_method)
        # UNCERTAINTY #
        self.weight_unc_brier_inv_score, self.weight_unc_expected_cal_inv_error = self.load_weights_uncertainty(yaml_config_file)
        self.score_unc_brier_inv_score, self.score_unc_expected_cal_inv_error, self.score_uncertainty = self.compute_score_uncertainty()
        # TRUST #
        self.score_trust = self.compute_trust()
    
    def compute_trust(self):
        """Adds the Trust weighted factors and returns the resulting assessed Trust"""
        return self.score_performance+self.score_fairness+self.score_robustness+self.score_explainability+self.score_uncertainty+self.score_feedback
        
    def compute_score_feedback(self):
        """Computes the weighted feedback"""
        return self.trust_instance.feedback*self.weight_feedback
        
    def load_weights_performance(self, yaml_config_file):
        """Retrieves the weights for the performance metrics"""
        weight_perf_accuracy = yaml_config_file['w.avg_parameters']['performance']['perf_metrics']['accuracy_weight']
        weight_perf_precision = yaml_config_file['w.avg_parameters']['performance']['perf_metrics']['precision_weight']
        weight_perf_recall = yaml_config_file['w.avg_parameters']['performance']['perf_metrics']['recall_weight']
        weight_perf_f1 = yaml_config_file['w.avg_parameters']['performance']['perf_metrics']['f1_weight']

        if weight_perf_accuracy + weight_perf_precision + weight_perf_recall + weight_perf_f1 != 1:
            raise ValueError('Performance metrics\' weights do not add 1. Revise the configuration file')
        else:
            return weight_perf_accuracy, weight_perf_precision, weight_perf_recall, weight_perf_f1

    def compute_score_performance(self):
        """Computes the weighted performance"""
        score_perf_accuracy = self.trust_instance.perf_accuracy * self.weight_perf_accuracy
        score_perf_precision = self.trust_instance.perf_precision * self.weight_perf_precision
        score_perf_recall = self.trust_instance.perf_recall * self.weight_perf_recall
        score_perf_f1 = self.trust_instance.perf_f1 * self.weight_perf_f1

        score_performance = (score_perf_accuracy+score_perf_precision+score_perf_recall+score_perf_f1)*self.weight_performance

        return score_perf_accuracy, score_perf_precision, score_perf_recall, score_perf_f1, score_performance
        
    def load_weights_fairness(self, yaml_config_file):
        """Retrieves the weights for the fairness metrics"""
        weight_fair_p_percentage = yaml_config_file['w.avg_parameters']['fairness']['fair_metrics']['p-percentage_weight']
        weight_fair_equal_opportunity_score = yaml_config_file['w.avg_parameters']['fairness']['fair_metrics']['equal_opportunity_weight']

        if weight_fair_p_percentage + weight_fair_equal_opportunity_score != 1:
            raise ValueError('Fairness metrics\' weights do not add 1. Revise the configuration file')
        else:
            return weight_fair_p_percentage, weight_fair_equal_opportunity_score

    def compute_score_fairness(self):
        """Computes the weighted fairness"""
        score_fair_p_percentage = self.trust_instance.fair_p_percentage * self.weight_fair_p_percentage
        score_fair_equal_opportunity = self.trust_instance.fair_equal_opportunity_score * self.weight_fair_equal_opportunity_score

        score_fairness = (score_fair_p_percentage+score_fair_equal_opportunity)*self.weight_fairness

        return score_fair_p_percentage, score_fair_equal_opportunity, score_fairness

    def load_weights_robustness(self, yaml_config_file):
        """Retrieves the weights for the robustness metrics"""
        weight_rob_average_bound = yaml_config_file['w.avg_parameters']['robustness']['rob_metrics']['average_bound_weight']
        weight_rob_verified_inv_error = yaml_config_file['w.avg_parameters']['robustness']['rob_metrics']['verified_error_inv_weight']

        if weight_rob_average_bound + weight_rob_verified_inv_error != 1:
            raise ValueError('Robustness metrics\' weights do not add 1. Revise the configuration file')
        else:
            return weight_rob_average_bound, weight_rob_verified_inv_error

    def compute_score_robustness(self):
        """Computes the weighted robustness"""
        score_rob_average_bound = self.trust_instance.rob_average_bound * self.weight_rob_average_bound
        score_rob_verified_inv_error = self.trust_instance.rob_verified_inv_error * self.weight_rob_verified_inv_error

        score_robustness = (score_rob_average_bound+score_rob_verified_inv_error)*self.weight_robustness

        return score_rob_average_bound, score_rob_verified_inv_error, score_robustness
    
    def load_weights_score_explainability_LIME(self, yaml_config_file):
        """Retrieves the weights for the explainability metrics when using a LIME explainer"""
        self.weight_expl_average_monotonicity = yaml_config_file['w.avg_parameters']['explainability']['expl_metrics_LIME']['average_monotonicity_weight']
        self.weight_expl_average_faithfulness = yaml_config_file['w.avg_parameters']['explainability']['expl_metrics_LIME']['average_faithfulness_weight'] 

        if self.weight_expl_average_monotonicity + self.weight_expl_average_faithfulness != 1:
            raise ValueError('Explainability metrics\' weights (LIME) do not add 1. Revise the configuration file')
        else:            
            self.score_expl_average_monotonicity = self.trust_instance.expl_average_monotonicity * self.weight_expl_average_monotonicity
            self.score_expl_average_faithfulness = self.trust_instance.expl_average_faithfulness * self.weight_expl_average_faithfulness

            self.score_explainability = (self.score_expl_average_monotonicity+self.score_expl_average_faithfulness)*self.weight_explainability

    def load_weights_score_explainability_TED(self, yaml_config_file):
        """Retrieves the weights for the explainability metrics when using a TED-enhanced classifier and explanations"""
        self.weight_expl_E_accuracy_score = yaml_config_file['w.avg_parameters']['explainability']['expl_metrics_TED']['E_score_weight']

        if self.weight_expl_E_accuracy_score != 1:
            raise ValueError('Explainability metrics\' weights (TED) do not add 1. Revise the configuration file')
        else:
            self.score_expl_E_accuracy_score = self.trust_instance.expl_E_accuracy*self.weight_expl_E_accuracy_score

            self.score_explainability = self.score_expl_E_accuracy_score*self.weight_explainability    

    def load_weights_score_explainability(self, yaml_config_file, explainability_method):
        """Retrieves the weights for the explainability metrics"""
        if (explainability_method is Trust.Explainability_method.LIME):
            self.load_weights_score_explainability_LIME(yaml_config_file)
        else:
            self.load_weights_score_explainability_TED(yaml_config_file) 

    def load_weights_uncertainty(self, yaml_config_file):
        """Retrieves the weights for the uncertainty metrics"""
        weight_unc_brier_inv_score = yaml_config_file['w.avg_parameters']['uncertainty']['uncert_metrics']['brier_inv_score_weight']
        weight_unc_expected_cal_inv_error = yaml_config_file['w.avg_parameters']['uncertainty']['uncert_metrics']['expected_cal_inv_error_weight'] 

        if weight_unc_brier_inv_score + weight_unc_expected_cal_inv_error != 1:
            raise ValueError('Uncertainty metrics\' weights do not add 1. Revise the configuration file')
        else:
            return weight_unc_brier_inv_score, weight_unc_expected_cal_inv_error

    def compute_score_uncertainty(self):
        """Computes the weighted uncertainty"""
        score_unc_brier_inv_score = self.trust_instance.unc_brier_inv_score * self.weight_unc_brier_inv_score
        score_unc_expected_cal_inv_error = self.trust_instance.unc_expected_cal_inv_error * self.weight_unc_expected_cal_inv_error

        score_uncertainty = (score_unc_brier_inv_score+score_unc_expected_cal_inv_error)*self.weight_uncertainty

        return score_unc_brier_inv_score, score_unc_expected_cal_inv_error, score_uncertainty

    def __str__(self):
        """Returns a string with the assessed Trust along with its assessed factors"""
        return "TRUST EVALUATED AS A WEIGHTED AVERAGE: {:.2f}".format(self.score_trust) + "\nWeighted Performance: {:.2f}".format(self.score_performance) + \
            "\nWeighted Fairness: {:.2f}".format(self.score_fairness) + "\nWeighted Uncertainty: {:.2f}".format(self.score_uncertainty) \
                + "\nWeighted Explainability: {:.2f}".format(self.score_explainability) + "\nWeighted Robustness: {:.2f}".format(self.score_robustness) \
                    + "\nWeighted Feedback: {:.2f}".format(self.score_feedback)

    def get_Trust_WA_as_JSON(self):
        """Returns the weighted and assessed trust along with its components and used weights as a JSON"""
        # Create Dictionary
        trust_assessment = {
            "ABOUT": "TRUST AND ITS COMPONENTS ASSESSED AS A WEIGHTED AVERAGE",
            "TRUST": self.score_trust,
            "feedback":
                {
                    "weight": self.weight_feedback,
                    "score": self.score_feedback
                },
            "performance":
                {
                    "weight": self.weight_performance,
                    "score": self.score_performance,
                    "accuracy":
                        {
                            "weight": self.weight_perf_accuracy,
                            "score": self.score_perf_accuracy
                        },                        
                    "precision":
                        {
                            "weight": self.weight_perf_precision,
                            "score":  self.score_perf_precision
                        },
                    "recall":
                        {
                            "weight": self.weight_perf_recall,
                            "score": self.score_perf_recall
                        }
                },
            "fairness":
                {
                    "weight": self.weight_fairness,
                    "score": self.score_fairness,
                    "p_percentage":
                        {
                            "weight": self.weight_fair_p_percentage,
                            "score": self.score_fair_p_percentage
                        },
                    "equal_opportunity":
                        {
                            "weight": self.weight_fair_equal_opportunity_score,
                            "score": self.score_fair_equal_opportunity
                        }
                },
            "robustness":
                {
                    "weight": self.weight_robustness,
                    "score": self.score_robustness,
                    "average_bound":
                        {
                            "weight": self.weight_rob_average_bound,
                            "score": self.score_rob_average_bound
                        },
                    "verified_error_inv":
                        {
                            "weight": self.weight_rob_verified_inv_error,
                            "score": self.score_rob_verified_inv_error
                        }
                },
            "explainability":
                {
                    "weight": self.weight_explainability,
                    "score": self.score_explainability,
                    "average_monotonicity_LIME":
                        {
                            "weight": self.weight_expl_average_monotonicity,
                            "score": self.score_expl_average_monotonicity
                        },
                    "average_faithfulness_LIME":
                        {
                            "weight": self.weight_expl_average_faithfulness,
                            "score": self.score_expl_average_faithfulness
                        },
                    "E_score_TED":
                        {
                            "weight": self.weight_expl_E_accuracy_score,
                            "score": self.score_expl_E_accuracy_score
                        }                    
                },
            "uncertainty":
                {
                    "weight": self.weight_uncertainty,
                    "score": self.score_uncertainty,
                    "brier_score":
                        {
                            "weight": self.weight_unc_brier_inv_score,
                            "score": self.score_unc_brier_inv_score
                        },
                    "expected_calibration_error_inv":
                        {
                            "weight": self.weight_unc_expected_cal_inv_error,
                            "score": self.score_unc_expected_cal_inv_error
                        }
                    
                }
        }
        return json.dumps(trust_assessment, indent=4)