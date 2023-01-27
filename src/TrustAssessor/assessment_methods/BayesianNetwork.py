import requests
from TrustAssessor.assessment_methods.AssessmentMethod import AssessmentMethod
from json import dumps

class BayesianNetwork(AssessmentMethod):
    """Class that evaluates a TrustEntity using a Bayesian network, by using a BN model provided in the path specified in the YAML/JSON configuration file.
    It requires an instance of a TrustableEntity, which contains the trust metrics already computed and unweighted.

    It also requires an active and listening server with the SSI-Assessment API-library deployed (https://github.com/martimanzano/SSI-assessment).
    The endpoint shall be specified in the YAML/JSON configuration.
    """

    def __init__(self, additionalProperties):
        """Retrieves the BN path and discretization intervals from the loaded YAML/JSON and prepares the object's attributes to
        assess trust and its related components, using the trustable entity received as parameter.

        Args:
            trustable_entity (TrustableEntity): [Instanced and initialized TrustableEntity object with unweighted metrics computed]
        
        Raises:
            Exception: When assessed metrics and BN's binning intervals are not consistent
        """

        super().__init__()

        self.BNPath = additionalProperties['bn_path']
        self.APIAssessmentService = additionalProperties['api_url']
        self.IDTrustNode = additionalProperties['id_trust_node']

        self.inputNames = [k for d in additionalProperties['intervals_input_nodes'] for k in d.keys()]
        self.intervalsInputNodes = [k for d in additionalProperties['intervals_input_nodes'] for k in d.values()]
       
    def assess(self):
        """Calls the BN assessment service synchrounously to assess the Trust/Trust factor. Stores the response in the instance as a JSON"""

        if not self.compareConfigAssesssedMetricsInputs():
            raise Exception("Validation error in config file: assessed metrics and BN's input binning intervals mismatch")

        inputValues = []
        for inputName in self.inputNames:
            inputValues.append(self.getMetricsAssessmentDict()[inputName])
                   
            apiResponse = requests.post(url=self.APIAssessmentService, 
            json={'id_si': self.IDTrustNode, 'input_names': self.inputNames,'input_values': inputValues, 'intervals_input_nodes': self.intervalsInputNodes, 'bn_path': self.BNPath})
        
        self.assessmentObject = apiResponse.json()
        self.assessment = dumps(self.assessmentObject)

    def compareConfigAssesssedMetricsInputs(self):
        """Helper function to validate the binning intervals from the configuration dict.

        Returns:
            Boolean: True if binning intervals are consistent with the assessed metrics, False otherwise
        """
        assessedMetricsList = [metric.__class__.__name__ for metric in self.metrics]
    
        if set(self.inputNames).issubset(set(assessedMetricsList)):
            return True
        return False