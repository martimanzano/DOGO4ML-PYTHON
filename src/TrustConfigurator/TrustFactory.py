from TrustConfigurator.ConfigurationReader import ConfigurationReader
from TrustAssessor import assessment_methods, metrics
from TrustAssessor.assessment_methods.AssessmentMethod import AssessmentMethod
from TrustAssessor.metrics import *

_instances = {}

class Singleton(object):

   def __new__(cls, *args, **kw):
      if not cls in _instances:
          instance = super().__new__(cls)
          _instances[cls] = instance

      return _instances[cls]

class TrustFactory(Singleton):
    def __init__(self, configPath):
        self.configurationReader = ConfigurationReader(configPath=configPath)

    def createMetricsAndAssessmentMethod(self, trainedModel, dataX, dataY) -> AssessmentMethod:
        metricsToAssess = self.configurationReader.readMetricsFromFile()
        assessmentMethodNameAndProperties = self.configurationReader.readAssessmentMethodFromFile()

        assessmentMethodName = list(assessmentMethodNameAndProperties.keys())[0]
        assessmentMethodProperties = assessmentMethodNameAndProperties[assessmentMethodName]
        assessmentMethodInstance = getattr(assessment_methods, assessmentMethodName)(assessmentMethodProperties)#({k:v for d in assessmentMethodProperties for k, v in d.items()})

        for idx, metric in enumerate(metricsToAssess):
            if type(metric) is str:
                metricClass = getattr(metrics, metric)
                metricInstance = metricClass()
            else:
                metricClass = getattr(metrics, list(metric.keys())[0])
                metricInstance = metricClass(list(metric.values())[0])
            metricInstance.assess(trainedModel, dataX, dataY)
            assessmentMethodInstance.metrics.append(metricInstance)

        return assessmentMethodInstance