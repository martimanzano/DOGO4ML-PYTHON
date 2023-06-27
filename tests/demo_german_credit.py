from os import path
from dill import dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
from trustML.TrustworthinessComputationFacane.TrustworthinessComputationFacane import TrustworthinessComputationFacane

pd.reset_option("max_columns")
pd.set_option('display.max_columns', None)

demo_path = 'demos/german_credit/'
file_dataset = demo_path + 'german.data'
path_configuration_bank = demo_path + 'config_german_credit_banker_corporate.yml'

# DATASET HEADER
header = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount', 
         'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors', 
         'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing', 
         'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification']
label_column = 'classification'

numerical_variables = ['duration', 'creditamount', 'installmentrate', 'residencesince', 'age', 'existingcredits', 'peopleliable', 'classification']
categorical_variables = ['existingchecking', 'credithistory', 'purpose', 'savings', 'employmentsince',
           'statussex', 'otherdebtors', 'property', 'otherinstallmentplans', 'housing', 'job', 
           'telephone', 'foreignworker']

# protected_variables = ['statussex_A91', 'statussex_A92', 'statussex_A93', 'statussex_A94']

# DATASET LOAD
pd_dataset = pd.read_csv(file_dataset, names=header, delimiter= ' ')
#print(pd_dataset.shape)
#print(pd_dataset.columns)
pd_dataset.head(10)

# CLASSIFICATION COLUMN. [1,2] -> [1,0]
pd_dataset.classification.replace([1,2], [1,0], inplace=True)
pd_dataset.classification.value_counts()

# LABEL ENCODING FOR CATEGORICAL/ORDINAL FEATURES
pd_dataset = pd.get_dummies(pd_dataset)
#print(pd_dataset.columns)

# TRAIN-TEST SPLIT
# SPLIT FEATURES-TARGET (X-Y)
data_x = pd_dataset.drop(label_column, axis=1)
data_y = pd_dataset[label_column]
# SPLIT DATA
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.3, random_state=2) # TODO AMPLIAR VALIDATION SET A 0.3. SE REDUCE PARA ACELERAR LAS PRUEBAS

# TRAIN A RANDOM FOREST CLASSIFIER AND CREATE A TABULAR EXPLAINER
rf_classifier = RandomForestClassifier(max_depth=10, n_estimators=100,criterion='gini')
rf_model = rf_classifier.fit(train_x, train_y)
lime_explainer = LimeTabularExplainer(train_x.values, feature_names=data_x.columns.values, class_names=[1,0])

# SAVE EXPLAINER TO THE DEMO DIRECTORY (OPTIONAL, ONLY NECESSARY IF THEY ARE NOT ALREADY PRESENT)
with open(demo_path + 'lime_explainer', 'wb') as explainer_file:
    dump(lime_explainer, explainer_file)

# TRUST STUFF
trustBank = TrustworthinessComputationFacane()
trustBank.loadTrustworthinessIndicator(configPath=path_configuration_bank)
trustBank.computeTrustworthinessScore(trainedModel=rf_model, dataX=test_x, dataY=test_y)

print(trustBank.getTrustworthinessScore())