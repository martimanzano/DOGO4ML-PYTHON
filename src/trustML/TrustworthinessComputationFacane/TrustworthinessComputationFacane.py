from trustML.TrustworthinessSpecification.TrustFactory import TrustFactory

class TrustworthinessComputationFacane():
    """Class that provides the package's functionality to the endusers/systems.
    """
    def __init__(self):
        self.TWI = None

    def loadTrustworthinessIndicator(self, configPath):
        """Loads and inicializes the required components (metrics, assessment
        method) from the provided configuration file.

        Args:
            configPath (String): Path to the configuration file
        """
        trustFactory = TrustFactory(configPath)
        self.TWI = trustFactory.createTWI()

    def computeTrustworthinessScore(self, trainedModel, dataX, dataY):
        """Performs the metrics' assessments, followed by the trust assessment
        based on such metrics and the specified assessment method and its parameters.
        Leverages this process to the TWI class.

        Args:
            trainedModel (sklearn-based classifier): classifier to evaluate
            dataX (pandas dataset): predictor data of the dataset to evaluate
            dataY (pandas dataset): target values of the dataset to evaluate
        """
        
        self.TWI.assess(trainedModel, dataX, dataY)

    def getTrustworthinessScore(self) -> str:
        """Returns the trust assessment as a formated JSON string.

        Returns:
            str: trustworthiness assessment
        """
        return self.TWI.getTWS()