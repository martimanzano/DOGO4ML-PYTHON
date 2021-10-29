import Trust

class Trust_Weighted_Average():
    def __init__(self, trust_instance : Trust):
        self.trust_instance = trust_instance    
        self.load_weights()
        self.compute_trust()

    def compute_trust(self):
        self.score_trust_without_feedback = self.score_performance+self.score_fairness+self.score_robustness+self.score_explainability
        self.score_trust_with_feedback = self.score_trust_without_feedback*self.trust_instance.feedback

    def load_weights(self):
        self.weight_performance = self.trust_instance.yaml_config_file['performance']['weight']
        self.weight_fairness = self.trust_instance.yaml_config_file['fairness']['weight']
        self.weight_robustness = self.trust_instance.yaml_config_file['robustness']['weight']
        self.weight_explainability = self.trust_instance.yaml_config_file['explainability']['weight']

        if self.weight_performance + self.weight_fairness + self.weight_robustness + self.weight_explainability != 1:
            raise ValueError('Factor weights do not add 1. Revise the configuration file')
        self.load_weights_performance()
        self.load_weights_fairness()
        self.load_weights_robustness()
        self.load_weights_explainability()
    
    def load_weights_performance(self):
        self.weight_perf_accuracy = self.trust_instance.yaml_config_file['performance']['perf_metrics']['accuracy_weight']
        self.weight_perf_precision = self.trust_instance.yaml_config_file['performance']['perf_metrics']['precision_weight']
        self.weight_perf_recall = self.trust_instance.yaml_config_file['performance']['perf_metrics']['recall_weight']
        self.weight_perf_f1 = self.trust_instance.yaml_config_file['performance']['perf_metrics']['f1_weight']

        if self.weight_perf_accuracy + self.weight_perf_precision + self.weight_perf_recall + self.weight_perf_f1 != 1:
            raise ValueError('Performance metrics\' weights do not add 1. Revise the configuration file')
        else:
            self.score_perf_accuracy = self.trust_instance.perf_accuracy * self.weight_perf_accuracy
            self.score_perf_precision = self.trust_instance.perf_precision * self.weight_perf_precision
            self.score_perf_recall = self.trust_instance.perf_recall * self.weight_perf_recall
            self.score_perf_f1 = self.trust_instance.perf_f1 * self.weight_perf_f1

            self.score_performance = (self.score_perf_accuracy+self.score_perf_precision+self.score_perf_recall+self.score_perf_f1)*self.weight_performance
        
    def load_weights_fairness(self):
        self.weight_fair_p_percentage = self.trust_instance.yaml_config_file['fairness']['fair_metrics']['p-percentage_weight']

        if self.weight_fair_p_percentage != 1:
            raise ValueError('Fairness metrics\' weights do not add 1. Revise the configuration file')
        else:
            self.score_fair_p_percentage = self.trust_instance.fair_p_percentage * self.weight_fair_p_percentage

            self.score_fairness = (self.score_fair_p_percentage)*self.weight_fairness

    def load_weights_robustness(self):
        self.weight_rob_average_bound = self.trust_instance.yaml_config_file['robustness']['rob_metrics']['average_bound_weight']
        self.weight_rob_verified_inv_error = self.trust_instance.yaml_config_file['robustness']['rob_metrics']['verified_error_inv_weight']

        if self.weight_rob_average_bound + self.weight_rob_verified_inv_error != 1:
            raise ValueError('Robustness metrics\' weights do not add 1. Revise the configuration file')
        else:
            self.score_rob_average_bound = self.trust_instance.rob_average_bound * self.weight_rob_average_bound
            self.score_rob_verified_inv_error = self.trust_instance.rob_verified_inv_error * self.weight_rob_verified_inv_error

            self.score_robustness = (self.score_rob_average_bound+self.score_rob_verified_inv_error)*self.weight_robustness

    def load_weights_explainability(self):
        self.weight_expl_average_monotonicity = self.trust_instance.yaml_config_file['explainability']['expl_metrics']['average_monotonicity_weight']
        self.weight_expl_average_faithfulness = self.trust_instance.yaml_config_file['explainability']['expl_metrics']['average_faithfulness_weight'] 

        if self.weight_expl_average_monotonicity + self.weight_expl_average_faithfulness != 1:
            raise ValueError('Explainability metrics\' weights do not add 1. Revise the configuration file')
        else:
            self.score_expl_average_monotonicity = self.trust_instance.expl_average_monotonicity * self.weight_expl_average_monotonicity
            self.score_expl_average_faithfulness = self.trust_instance.expl_average_faithfulness * self.weight_expl_average_faithfulness

            self.score_explainability = (self.score_expl_average_monotonicity+self.score_expl_average_faithfulness)*self.weight_explainability