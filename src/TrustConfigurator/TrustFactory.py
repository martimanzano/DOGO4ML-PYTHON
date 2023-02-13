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
    """Singleton class in charge of reading the configuration file through 
    an instantiated ConfigurationReader and performing the instantiations 
    of the metric instances to be assessed and their associations with the
    specified assessment method, which is also instantiated.

    Args:
        Singleton (Class): Singleton implementation for Python
    """
    def __init__(self, configPath):
        """Instantiates a ConfigurationReader object to retrieve the required
        data from the configuration file specified.

        Args:
            configPath (str): Filepath to the configuration file
        """
        self.configurationReader = ConfigurationReader(configPath=configPath)

    def createMetricsAndAssessmentMethod(self) -> AssessmentMethod:
        """Instantiates the metric instances and the assessment method to be used for the trust assessment. Associates the set of metrics to such assessment method, which is returned.

        Returns:
            AssessmentMethod: Instance of the asessment method with the metrics
            to be used for the trust assessment
        """
        metricsToAssess = self.configurationReader.readMetricsFromFile()
        assessmentMethodNameAndProperties = self.configurationReader.readAssessmentMethodFromFile()

        assessmentMethodName = list(assessmentMethodNameAndProperties.keys())[0]
        assessmentMethodProperties = assessmentMethodNameAndProperties[assessmentMethodName]
        assessmentMethodInstance = getattr(assessment_methods, assessmentMethodName)(assessmentMethodProperties)
        #({k:v for d in assessmentMethodProperties for k, v in d.items()})

        for idx, metric in enumerate(metricsToAssess):
            if type(metric) is str:
                metricClass = getattr(metrics, metric)
                metricInstance = metricClass()
            else:
                metricClass = getattr(metrics, list(metric.keys())[0])
                metricInstance = metricClass(list(metric.values())[0])            
            assessmentMethodInstance.metrics.append(metricInstance)

        return assessmentMethodInstance