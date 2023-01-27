from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from aix360.algorithms.ted.TED_Cartesian import TED_CartesianExplainer
from TrustComputationService.TrustComputationService import TrustComputationService
import pandas as pd

pd.reset_option("max_columns")
pd.set_option('display.max_columns', None)

dataset_path = 'Retention.csv'

configuration_path = 'configs/trust_config_retention_corporate - refactor.yml'

# Decompose the dataset into X, Y, E
data = pd.read_csv(dataset_path)
X = data.iloc[:,:-2]   # Choose all rows and all cols, except for the last 2 cols
Y = data['Y']          # Choose col with header 'Y'
E = data['E']          # Choose col with header 'E'

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

# TRUST STUFF
trust_instance = TrustComputationService()
trust_instance.computeTrust(configuration_path, ted, X_test, Y_test)
print(trust_instance.getTrustAssessment())