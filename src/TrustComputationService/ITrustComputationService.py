class ITrustComputationService:
    def __init__(self):
        pass

    def computeTrust(self, configPath, trainedModel, dataX, dataY) -> str:
        pass

    def getTrustAssessment(self) -> str:
        pass

    def getTrustAssessmentAsObject(self):
        pass
    
    def getMetricAssessment(self, metricName) -> float:
        pass

    def getMetricsAssessments(self) -> dict:
        pass