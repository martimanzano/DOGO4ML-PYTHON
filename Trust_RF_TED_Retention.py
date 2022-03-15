from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from aix360.algorithms.ted.TED_Cartesian import TED_CartesianExplainer
from aix360.datasets.ted_dataset import TEDDataset
from trust.Trust import Trust, Evaluation_method
import pandas as pd

pd.reset_option("max_columns")
pd.set_option('display.max_columns', None)

file_dataset = 'Retention.csv'

yaml_configuration_corp = 'configs/trust4RF_config_retention_corporate.yml'
yaml_configuration_HR = 'configs/trust4RF_config_retention_HR.yml'

yaml_configuration_to_use = yaml_configuration_corp

# Decompose the dataset into X, Y, E     
X, Y, E = TEDDataset().load_file(file_dataset)

# CLASSIFICATION COLUMN. [-10,-11] -> [0,1]
Y.replace([-10,-11], [0,1], inplace=True)

# Set up train/test split
X_train, X_test, Y_train, Y_test, E_train, E_test = train_test_split(X, Y, E, test_size=0.01, random_state=1)

# TRAIN A RANDOM FOREST CLASSIFIER AND CREATE A TED-ENHANCED CLASSIFIER
rf_classifier = RandomForestClassifier()
ted = TED_CartesianExplainer(rf_classifier)

ted.fit(X_train, Y_train, E_train)
YE_accuracy, Y_accuracy, E_accuracy = ted.score(X_test, Y_test, E_test)    # evaluate the classifier
print("Evaluating accuracy of TED-enhanced classifier on test data")
print(' Accuracy of predicting Y labels: %.2f%%' % (100*Y_accuracy))
print(' Accuracy of predicting explanations: %.2f%%' % (100*E_accuracy))
print(' Accuracy of predicting Y + explanations: %.2f%%' % (100*YE_accuracy))

### TRUST COMPUTATION AND CACHE MANAGEMENT ###
trust = Trust.load_compute_trust_with_cache(file_dataset, yaml_configuration_to_use, X_test, Y_test, ted, lime_explainer=None, data_E=E_test)
trust_weighted_avrg = trust.evaluate_trust(yaml_configuration_to_use, Evaluation_method.Weighted_Average)

print(trust.get_Trust_metrics_as_JSON())
print(trust_weighted_avrg)
print(trust_weighted_avrg.get_Trust_WA_as_JSON())
#trust_bn = trust.evaluate_trust(yaml_configuration_to_use, Evaluation_method.Bayesian_Network)
# print("Trust: Bayesian network: " + repr(trust_bn.bn_assessment))