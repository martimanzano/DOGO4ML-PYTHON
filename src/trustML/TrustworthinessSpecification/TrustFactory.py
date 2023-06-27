#import trustML.TrustworthinessComputation
import yaml
from trustML.TrustworthinessComputation import assessment_methods, metrics
#from trustML.TrustworthinessComputation.assessment_methods.AssessmentMethod import AssessmentMethod
#from trustML.TrustworthinessComputation.metrics import *
from trustML.TrustworthinessComputation.TWI import TWI

_instances = {}

class Singleton(object):

   def __new__(cls, *args, **kw):
      if not cls in _instances:
          instance = super().__new__(cls)
          _instances[cls] = instance

      return _instances[cls]

class TrustFactory(Singleton):
    """Singleton class in charge of reading the configuration file through 
    a YAML reader and performing the initial management of the metric instances
    to be assessed and their associations with the specified assessment method,
    which is also instantiated. The association is performed through an instance
    of the TWI class.

    Args:
        Singleton (Class): Singleton implementation for Python
    """
    def __init__(self, configPath):
        """Instantiates a TrustFactory object to retrieve and stores the required
        data from the configuration file specified.

        Args:
            configPath (str): Filepath to the configuration file
        """
        with open(configPath, mode='r') as config_file:            
            parsedConfig = yaml.safe_load(config_file)
            self.metricsFromFile = parsedConfig['metrics']
            self.assessmentMethodFromFile = parsedConfig['assessment_method']

    def createTWI(self) -> TWI:
        """Instantiates the metric objects and the assessment method to be used for the trust assessment. 
        Associates the set of metrics and assessment method to a Trustworthiness Indicator (TWI), which is returned.

        Returns:
            TWI: Instance of the Trustworthiness Indicator with the instanced asessment method and metrics
            to be used for the trust computation
        """

        assessmentMethodName = list(self.assessmentMethodFromFile.keys())[0]
        assessmentMethodProperties = self.assessmentMethodFromFile[assessmentMethodName]
        assessmentMethodInstance = getattr(assessment_methods, assessmentMethodName)(assessmentMethodProperties)
        instancedMetrics = []
        #({k:v for d in assessmentMethodProperties for k, v in d.items()})

        for idx, metric in enumerate(self.metricsFromFile):
            if type(metric) is str:
                metricClass = getattr(metrics, metric)
                metricInstance = metricClass()
            else:
                metricClass = getattr(metrics, list(metric.keys())[0])
                metricInstance = metricClass(list(metric.values())[0])            
            instancedMetrics.append(metricInstance)

        ret = TWI()
        ret.metrics = instancedMetrics
        ret.assessmentMethod = assessmentMethodInstance
        ret.assessmentMethod.TWI = ret

        return ret