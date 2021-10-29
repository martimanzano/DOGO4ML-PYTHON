import Trust
import requests

class Trust_BN():
    def __init__(self, trust_instance : Trust):
        self.trust_instance = trust_instance
        self.load_bn_parameters()
        self.bn_assessment = self.assess_trust_bn()
        
            
    def load_bn_parameters(self):
        self.bn_path = self.trust_instance.yaml_config_file['general']['bn_parameters']['bn_path']
        self.api_url = self.trust_instance.yaml_config_file['general']['bn_parameters']['api_url']
        self.id_si = self.trust_instance.yaml_config_file['general']['bn_parameters']['id_si']

        self.input_names = [k for d in self.trust_instance.yaml_config_file['general']['bn_parameters']['intervals_input_nodes'] for k in d.keys()]
        self.intervals_input_nodes = [k for d in self.trust_instance.yaml_config_file['general']['bn_parameters']['intervals_input_nodes'] for k in d.values()]

    def assess_trust_bn(self):
        input_values = [self.trust_instance.perf_accuracy, self.trust_instance.perf_precision, \
            self.trust_instance.perf_recall, self.trust_instance.perf_f1, self.trust_instance.fair_p_percentage, \
                self.trust_instance.rob_average_bound, self.trust_instance.rob_verified_inv_error, self.trust_instance.expl_average_monotonicity, \
                    self.trust_instance.expl_average_faithfulness, self.trust_instance.feedback]
   
        with open(self.bn_path, 'rb') as f:
            api_response = requests.post(url=self.api_url,
                            json={'id_si': self.id_si, 'input_names': self.input_names,'input_values': input_values, 'intervals_input_nodes': self.intervals_input_nodes, 'bn_path': self.bn_path})
        return api_response.content