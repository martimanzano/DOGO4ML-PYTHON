from TrustConfigurator.TrustFactory import TrustFactory
from TrustComputationService.ITrustComputationService import ITrustComputationService

class TrustComputationService(ITrustComputationService):
    def __init__(self):
        self.assessmentMethod = None

    def computeTrust(self, configPath, trainedModel, dataX, dataY) -> str:
        trustFactory = TrustFactory(configPath)
        self.assessmentMethod = trustFactory.createMetricsAndAssessmentMethod(trainedModel, dataX, dataY)
        self.assessmentMethod.assess()

        return self.assessmentMethod.getAssessment()

    def getTrustAssessment(self) -> str:
        return self.assessmentMethod.getAssessment()

    def getTrustAssessmentAsObject(self):
        return self.assessmentMethod.getAssessmentObject()
    
    def getMetricAssessment(self, metricName) -> float:
        return self.getMetricsAssessments()[metricName]

    def getMetricsAssessments(self) -> dict:
        return self.assessmentMethod.getMetricsAssessmentDict()