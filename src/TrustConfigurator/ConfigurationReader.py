import yaml

class ConfigurationReader():
    
    def __init__(self, configPath):        
        with open(configPath, mode='r') as config_file:            
            self.parsedConfig = yaml.load(config_file)

    def readMetricsFromFile(self) -> dict:
        return self.parsedConfig['metrics']
    
    def readAssessmentMethodFromFile(self) -> dict:
        return self.parsedConfig['assessment_method']