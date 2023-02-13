import yaml

class ConfigurationReader():
    """Class used by the TrustFactory to read the configuration
    file and retrieve the required information for the trust
    assessment.
    """
    
    def __init__(self, configPath):
        """
        Stores the parsed yaml configuration file as a dictionary.
        """ 
        with open(configPath, mode='r') as config_file:            
            self.parsedConfig = yaml.safe_load(config_file)

    def readMetricsFromFile(self) -> dict:
        """Returns the data related to the metrics' assessment
        from the loaded configuration file

        Returns:
            dict: metrics' required data
        """
        return self.parsedConfig['metrics']
    
    def readAssessmentMethodFromFile(self) -> dict:
        """Returns the data related to the assessment
        method from the loaded configuration file

        Returns:
            dict: assessment method's required data
        """
        return self.parsedConfig['assessment_method']