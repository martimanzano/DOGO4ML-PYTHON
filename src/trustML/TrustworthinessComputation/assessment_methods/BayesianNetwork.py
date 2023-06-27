import requests
from trustML.TrustworthinessComputation.assessment_methods.AssessmentMethod import AssessmentMethod
from json import dumps

class BayesianNetwork(AssessmentMethod):
    """Class that implements the trust assessment using a Bayesian network in DNE format,
    by using a BN model previously crafted and provided in the filepath specified in the
    configuration file. It requires to have the trust metrics already computed in the TWI object.

    It also requires an active and listening server with the SSI-Assessment API-library 
    deployed (https://github.com/martimanzano/SSI-assessment). Its endpoint shall be 
    specified in the configuration file.
    """

    def __init__(self, additionalProperties):
        """Retrieves the set of parameters required to perform the assessment through a BN
        (including the BN filepath and discretization intervals) from the additional properties
        retrieved from the configuration file and prepares the instance's attributes to perform
        the trustworthiness assessment using the SSI assessment library.

        Args:
            additionalProperties (dict): [dictionary of parameters required by the assessment method,
            i.e., the BN's filepath, endpoint of the assessment service, 
            the BN node corresponding to the trustworthiness, and the discretization intervals to use]
        
        Raises:
            Exception: When assessed metrics and BN's binning intervals are not consistent
        """

        super().__init__()

        self.BNPath = additionalProperties['bn_path']
        self.APIAssessmentService = additionalProperties['api_url']
        self.IDTrustNode = additionalProperties['id_trust_node']

        self.inputNodes = additionalProperties['intervals_input_nodes']        
       
    def assess(self):
        """Calls the BN assessment service synchrounously to assess the BN node with name equal to the 
        "IDTrustNode" attribute. Returns the result as a JSON formatted string containing the node's 
        probabilitiies.
        """

        inputNames = [k for d in self.inputNodes for k in d.keys()]
        intervalsInputNodes = [k for d in self.inputNodes for k in d.values()]

        if not self.compareConfigAssesssedMetricsInputs(inputNames):
            raise Exception("Validation error in config file: assessed metrics and BN's input binning intervals mismatch")

        inputValues = []
        for inputName in inputNames:
            inputValues.append(self.TWI.getMetricsAssessmentDict()[inputName])
                   
            apiResponse = requests.post(url=self.APIAssessmentService, 
            json={'id_si': self.IDTrustNode, 'input_names': inputNames,'input_values': inputValues, 'intervals_input_nodes': intervalsInputNodes, 'bn_path': self.BNPath})
        
        assessment = apiResponse.json() 
        assessmentAsFormattedJSON = dumps(assessment)

        return assessmentAsFormattedJSON

    def compareConfigAssesssedMetricsInputs(self, inputNames):
        """Helper function to validate the binning intervals from the configuration dict.

        Returns:
            Boolean: True if binning intervals are consistent with the assessed metrics, False otherwise
        """
        assessedMetricsList = [metric.__class__.__name__ for metric in self.TWI.metrics]
    
        if set(inputNames).issubset(set(assessedMetricsList)):
            return True
        return False