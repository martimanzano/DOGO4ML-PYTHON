from pathlib import Path
from os import path
import trust.metrics
import yaml
import json
import dill

class TrustableEntity:
    """Class in charge of computing the trustworthiness metrics specified in the YAML/JSON configurations and loading their required
    parameters.
    """
    def __init__(self, config_path, trained_model, data_x, data_y, data_E = None, explainer = None, additional_properties = None):
        """Initializes the TrustableEntity with the attributes required to compute the trustworthiness metrics.

        Args:
            config_path (String): Path to the YAML/JSON configuration file
            trained_model (classification model): Pretrained classification model to be evaluated
            data_x (dataset): Independent variables to be used for the trustworthiness metrics' assessment
            data_y (dataset): Dependent variable (target column) to be used for the trustworthiness metrics' assessment
            data_E (dataset, optional): Explanations to be used for explainability-related metrics with TED explainers. Defaults to None.
            explainer (explainer, optional): Pretrained explainer to be used for explainability-related metrics. Defaults to None.
            additional_properties (dict, optional): Additional attributes that may be needed to compute new metrics. Defaults to None.
        """
        self.trained_model = trained_model
        self.data_x = data_x
        self.data_y = data_y
        self.data_E = data_E
        self.explainer = explainer
        self.additional_properties = additional_properties
        self.precomputed_metrics = {}
        self.config = self.load_config(config_path)
        self.protected_attributes = self.load_protected_attributes()        

        self.metrics_assessments = {}

    def load_config(self, config_path):
        """Loads the configuration file into a instance-stored dict

        Args:
            config_path (String): Path to the configuration file

        Raises:
            Exception: When the configuration file is not a YAML or JSON file

        Returns:
            dict: configuration dict
        """
        with open(config_path, mode='r') as config_file:
            file_extension = Path(config_path).suffix
            if file_extension == '.yml' or file_extension == '.yaml':
                return yaml.safe_load(config_file)
            elif file_extension == '.json':
                return json.load(config_file)
            else:
                raise Exception("Error: Configuration file's extension must be .yml, .yaml, or .json")
   
    def load_metrics_list(self):        
        return self.config['metrics']        

    def load_protected_attributes(self):      
        return self.config['protected_attributes']

    def assess_trust_metric(self, module, metric):
        """Assesses an individual metric from a specified module

        Args:
            module (module): module from which the metric will be evaluated
            metric (String): metric class from the module

        Returns:
            float: metric assessment
        """
        metric_class = getattr(module, metric)
        #metric_instance = metric_class()
        return metric_class.assess(self)        
        
    def assess_trust_metrics(self):
        """Traverses the trustworthiness metrics from the configuration dict and assesses each one
        """
        metrics_list = self.load_metrics_list()
        for metric in metrics_list:
            self.metrics_assessments[metric] = self.assess_trust_metric(trust.metrics, metric)

    def get_trust_metrics(self):
        """Getter for the trustworthiness metrics
        """
        return self.metrics_assessments

    def save_cache(self, directory):
        with open(directory, 'wb') as cached_trust_file:
            dill.dump(self, cached_trust_file)

    @staticmethod
    def load_cache(directory):
        if path.isfile(directory):
            with open(directory, 'rb') as cached_trust_file:
                return dill.load(cached_trust_file)