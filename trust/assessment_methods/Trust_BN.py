from trust.Trustable_entity import TrustableEntity
import requests
from trust.assessment_methods.assessment_interface import AssessmentInterface

class Trust_BN(AssessmentInterface):
    """Class that evaluates a TrustEntity using a Bayesian network, by using a BN model provided in the path specified in the YAML/JSON configuration file.
    It requires an instance of a TrustableEntity, which contains the trust metrics already computed and unweighted.

    It also requires an active and listening server with the SSI-Assessment API-library deployed (https://github.com/martimanzano/SSI-assessment).
    The endpoint shall be specified in the YAML/JSON configuration.
    """

    def __init__(self, trustable_entity : TrustableEntity):
        """Retrieves the BN path and discretization intervals from the loaded YAML/JSON and prepares the object's attributes to
        assess trust and its related components, using the trustable entity received as parameter.

        Args:
            trustable_entity (TrustableEntity): [Instanced and initialized TrustableEntity object with unweighted metrics computed]
        
        Raises:
            Exception: When assessed metrics and BN's binning intervals are not consistent
        """

        self.trustable_entity_instance = trustable_entity
        self.bn_path, self.api_url, self.id_si, self.input_names, self.intervals_input_nodes = self.load_bn_parameters()
        self.assessment = None
        if not self.compare_config_assesssed_metrics_inputs():
            raise("Validation error in config file: assessed metrics and BN's input binning intervals mismatch")
        
            
    def load_bn_parameters(self):
        """ Loads the required BN parameters from the configuration dict, i.e., path, API url of the BN assessement service, the Trust/Factor node
        to assess, and the BN's input names and discretization intervals.

        Returns:
            BN parameters loaded from the configuration dict
        """       
        
        yaml_config = self.trustable_entity_instance.config
        bn_path = yaml_config['bn_parameters']['bn_path']
        api_url = yaml_config['bn_parameters']['api_url']
        id_si = yaml_config['bn_parameters']['id_si']

        input_names = [k for d in yaml_config['bn_parameters']['intervals_input_nodes'] for k in d.keys()]
        intervals_input_nodes = [k for d in yaml_config['bn_parameters']['intervals_input_nodes'] for k in d.values()]

        return bn_path, api_url, id_si, input_names, intervals_input_nodes

    def assess(self):
        """Calls the BN assessment service synchrounously to assess the Trust/Trust factor. Stores the response in the instance as a JSON"""
        input_values = []
        for input_name in self.input_names:
            input_values.append(self.trustable_entity_instance.metrics_assessments[input_name])
           
        with open(self.bn_path, 'rb') as f:
            api_response = requests.post(url=self.api_url, 
            json={'id_si': self.id_si, 'input_names': self.input_names,'input_values': input_values, 'intervals_input_nodes': self.intervals_input_nodes, 'bn_path': self.bn_path})
        
        self.assessment = api_response.content

    def get_assessment(self):
        """Getter for the BN assessment variable"""
        return self.assessment

    def compare_config_assesssed_metrics_inputs(self):
        """Helper function to validate the binning intervals from the configuration dict.

        Returns:
            Boolean: True if binning intervals are consistent with the assessed metrics, False otherwise
        """
        assessed_metrics_list = self.trustable_entity_instance.load_metrics_list()
        bn_input_metrics_list = self.input_names
        if set(assessed_metrics_list) == set(bn_input_metrics_list):
            return True
        return False