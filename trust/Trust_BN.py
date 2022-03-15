import trust.Trust as Trust
import requests

class Trust_BN():
    """Class that evaluates the Trust using a Bayesian network, by using a BN model provided in the path specified in the yaml configuration file.
    It requires an instance of a Trust object, which contains the trust metrics already computed and unweighted.

    It also requires an active and listening server with the SSI-Assessment API-library deployed (https://github.com/martimanzano/SSI-assessment).
    The endpoint shall be specified in the yaml configuration file.
    """

    def __init__(self, trust_instance : Trust, yaml_config_file):
        """Retrieves the BN path and discretization intervals from the yaml configuration file and assesses the trust with its metrics and
        aggregations, using the Trust instance passed as parameter.

        Args:
            trust_instance (Trust): [Instanced and initialized Trust object with unweighted metrics computed]
            yaml_config_file ([yaml module object]): [yaml dict containing the user parameters]
        """

        self.trust_instance = trust_instance
        self.trust_instance.feedback = self.trust_instance.get_yaml_feedback(yaml_config_file)
        self.bn_path, self.api_url, self.id_si, self.input_names, self.intervals_input_nodes = self.load_bn_parameters(yaml_config_file)
        self.bn_assessment = self.assess_trust_bn()
            
    def load_bn_parameters(self, yaml_config_file):
        """Loads the required BN parameters from the yaml configuration file, i.e., path, API url of the BN assessement service, the Trust/Factor node
        to assess, and the BN's input names and discretization intervals."""
        bn_path = yaml_config_file['bn_parameters']['bn_path']
        api_url = yaml_config_file['bn_parameters']['api_url']
        id_si = yaml_config_file['bn_parameters']['id_si']

        input_names = [k for d in yaml_config_file['bn_parameters']['intervals_input_nodes'] for k in d.keys()]
        intervals_input_nodes = [k for d in yaml_config_file['bn_parameters']['intervals_input_nodes'] for k in d.values()]

        return bn_path, api_url, id_si, input_names, intervals_input_nodes

    def assess_trust_bn(self):
        """Calls the BN assessment service synchrounously to assess the Trust/Trust factor. Returns the response as a JSON"""
        input_values = [self.trust_instance.perf_accuracy, self.trust_instance.perf_precision, \
            self.trust_instance.perf_recall, self.trust_instance.perf_f1, self.trust_instance.fair_p_percentage, \
                self.trust_instance.rob_average_bound, self.trust_instance.rob_verified_inv_error, self.trust_instance.expl_average_monotonicity, \
                    self.trust_instance.expl_average_faithfulness, self.brier_inv_score, self.expected_cal_inv_error, self.trust_instance.feedback]
   
        with open(self.bn_path, 'rb') as f:
            api_response = requests.post(url=self.api_url,
                            json={'id_si': self.id_si, 'input_names': self.input_names,'input_values': input_values, 'intervals_input_nodes': self.intervals_input_nodes, 'bn_path': self.bn_path})
        
        return api_response.content

    def get_bn_assessment(self):
        """Getter for the BN assessment variable"""
        return self.bn_assessment