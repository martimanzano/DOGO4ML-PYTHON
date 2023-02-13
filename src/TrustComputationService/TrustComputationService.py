from TrustConfigurator.TrustFactory import TrustFactory
from TrustComputationService.ITrustComputationService import ITrustComputationService

class TrustComputationService(ITrustComputationService):
    """Class that provides the package's functionality to the endusers/systems.

    Args:
        ITrustComputationService (Class): Interface
    """
    def __init__(self):
        self.assessmentMethod = None

    def readTrustDefinition(self, configPath):
        """Loads and inicializes the required components (metrics, assessment
        method) from the provided configuration file.

        Args:
            configPath (String): Path to the configuration file
        """
        trustFactory = TrustFactory(configPath)
        self.assessmentMethod = trustFactory.createMetricsAndAssessmentMethod()

    def computeTrust(self, trainedModel, dataX, dataY):
        """Performs the metrics' assessments, followed by the trust assessment
        based on such metrics and the specified assessment method and its parameters.

        Args:
            trainedModel (sklearn classifier): classifier to evaluate
            dataX (pandas dataset): predictor data of the dataset to evaluate
            dataY (pandas dataset): target values of the dataset to evaluate
        """
        for metric in self.assessmentMethod.metrics:
            metric.assess(trainedModel, dataX, dataY)
        
        return self.assessmentMethod.assess()        

    def getTrustAssessmentAsFormattedString(self) -> str:
        """Returns the trust assessment as a string.

        Returns:
            str: trust assessment
        """
        return self.assessmentMethod.getAssessmentAsFormattedString()

    def getTrustAssessment(self):
        """Returns the trust assessment as the raw object produced by the
        assessment method

        Returns:
            obj: trust assessment
        """
        return self.assessmentMethod.getAssessment()
    
    def getMetricAssessment(self, metricName) -> float:
        """Returns the assessment of the specified metric.

        Args:
            metricName (str): metric name

        Returns:
            float: metric assessment
        """
        return self.getMetricsAssessments()[metricName]

    def getMetricsAssessments(self) -> dict:
        """Returns the complete set of assessed metrics as a dictionary

        Returns:
            dict: metrics' assessment (metric name -> metric assessment)
        """
        return self.assessmentMethod.getMetricsAssessmentDict()