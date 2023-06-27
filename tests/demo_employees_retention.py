from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from aix360.algorithms.ted.TED_Cartesian import TED_CartesianExplainer
from trustML.TrustworthinessComputationFacane.TrustworthinessComputationFacane import TrustworthinessComputationFacane
import pandas as pd
from pickle import dump

pd.reset_option("max_columns")
pd.set_option('display.max_columns', None)

demo_path = 'demos/employees_retention/'
dataset_path = demo_path + 'Retention.csv'
configuration_path = demo_path + 'config_retention_corporate.yml'

# Decompose the dataset into X, Y, E
data = pd.read_csv(dataset_path)
X = data.iloc[:,:-2]   # Choose all rows and all cols, except for the last 2 cols
Y = data['Y']          # Choose col with header 'Y'
E = data['E']          # Choose col with header 'E'

# CLASSIFICATION COLUMN. [-10,-11] -> [0,1]
Y.replace([-10,-11], [0,1], inplace=True)

# Set up train/test split
X_train, X_test, Y_train, Y_test, E_train, E_test = train_test_split(X, Y, E, test_size=0.3, random_state=1)

# TRAIN A RANDOM FOREST CLASSIFIER AND CREATE A TED-ENHANCED CLASSIFIER
rf_classifier = RandomForestClassifier()
ted = TED_CartesianExplainer(rf_classifier)

rf_classifier_raw = RandomForestClassifier().fit(X_train, Y_train)
ted.fit(X_train, Y_train, E_train)

YE_accuracy, Y_accuracy, E_accuracy = ted.score(X_test, Y_test, E_test)    # evaluate the classifier
print("Evaluating accuracy of TED-enhanced classifier on test data")
print(' Accuracy of predicting Y labels: %.2f%%' % (100*Y_accuracy))
print(' Accuracy of predicting explanations: %.2f%%' % (100*E_accuracy))
print(' Accuracy of predicting Y + explanations: %.2f%%' % (100*YE_accuracy))

# SAVE EXPLAINER AND EXPLANATIONS TO THE DEMO DIRECTORY (OPTIONAL, ONLY NECESSARY IF THEY ARE NOT ALREADY PRESENT)
with open(demo_path + 'ted_explainer', 'wb') as explainer_file:
    dump(ted, explainer_file)
with open(demo_path + 'ted_explanations', 'wb') as explanations_file:
    dump(E_test, explanations_file)

# TRUST STUFF
trust_instance = TrustworthinessComputationFacane()
trust_instance.loadTrustworthinessIndicator(configuration_path)
trust_instance.computeTrustworthinessScore(rf_classifier_raw, X_test, Y_test)

print(trust_instance.getTrustworthinessScore())
