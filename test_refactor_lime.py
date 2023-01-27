from os import path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
from TrustComputationService.TrustComputationService import TrustComputationService

pd.reset_option("max_columns")
pd.set_option('display.max_columns', None)
file_dataset = 'german.data'
path_configuration_to_use = 'configs/trust_config_german_credit_banker - refactor.yml'

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

protected_variables = ['statussex_A91', 'statussex_A92', 'statussex_A93', 'statussex_A94']

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
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.01, random_state=2) # TODO AMPLIAR VALIDATION SET A 0.3. SE REDUCE PARA ACELERAR LAS PRUEBAS

# TRAIN A RANDOM FOREST CLASSIFIER AND CREATE A TABULAR EXPLAINER
rf_classifier = RandomForestClassifier(max_depth=10, n_estimators=100,criterion='gini')
rf = rf_classifier.fit(train_x, train_y)
lime_explainer = LimeTabularExplainer(train_x.values, feature_names=data_x.columns.values, class_names=[1,0])

# TRUST STUFF
trust_instance = TrustComputationService()
trust_instance.computeTrust(path_configuration_to_use, rf, test_x, test_y)
print(trust_instance.getTrustAssessment())