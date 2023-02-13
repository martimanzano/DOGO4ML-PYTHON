import requests
from TrustAssessor.assessment_methods.AssessmentMethod import AssessmentMethod
from json import dumps

class BayesianNetwork(AssessmentMethod):
    """Class that implements the trust assessment using a Bayesian network in DNE format, by using a BN model previously crafted and provided in the filepath specified in the configuration file.
    It requires to have the trust metrics already computed and unweighted.

    It also requires an active and listening server with the SSI-Assessment API-library deployed (https://github.com/martimanzano/SSI-assessment).
    The endpoint shall be specified in the configuration file.
    """

    def __init__(self, additionalProperties):
        """Retrieves the BN path and discretization intervals from the additional properties retrieved from the configuration file and prepares the instance's attributes to perform the trust assessment.

        Args:
            additionalProperties (dict): [dictionary of parameters required by the assessment method, i.e., the BN's filepath, endpoint of the assessment service, the BN node to assess, and the discretization intervals to use]
        
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
        """Calls the BN assessment service synchrounously to assess the BN node with name equal to the "IDTrustNode" attribute. Stores the result as a JSON formatted string and as a dict containing the node's probabilitiies """

        if not self.compareConfigAssesssedMetricsInputs():
            raise Exception("Validation error in config file: assessed metrics and BN's input binning intervals mismatch")

        inputValues = []
        for inputName in self.inputNames:
            inputValues.append(self.getMetricsAssessmentDict()[inputName])
                   
            apiResponse = requests.post(url=self.APIAssessmentService, 
            json={'id_si': self.IDTrustNode, 'input_names': self.inputNames,'input_values': inputValues, 'intervals_input_nodes': self.intervalsInputNodes, 'bn_path': self.BNPath})
        
        self.assessment = apiResponse.json() 
        self.assessmentAsFormattedString = dumps(self.assessment)

        return self.assessment['probsSICategories']

    def compareConfigAssesssedMetricsInputs(self):
        """Helper function to validate the binning intervals from the configuration dict.

        Returns:
            Boolean: True if binning intervals are consistent with the assessed metrics, False otherwise
        """
        assessedMetricsList = [metric.__class__.__name__ for metric in self.metrics]
    
        if set(self.inputNames).issubset(set(assessedMetricsList)):
            return True
        return False