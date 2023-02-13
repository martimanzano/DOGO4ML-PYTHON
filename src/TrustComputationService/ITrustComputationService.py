class ITrustComputationService:
    """TrustComputationService interface.
    """
    def __init__(self):
        pass

    def readTrustDefinition(self, configPath):
        pass

    def computeTrust(self, trainedModel, dataX, dataY):
        pass

    def getTrustAssessmentAsFormattedString(self) -> str:
        pass

    def getTrustAssessment(self):
        pass
    
    def getMetricAssessment(self, metricName) -> float:
        pass

    def getMetricsAssessments(self) -> dict:
        pass