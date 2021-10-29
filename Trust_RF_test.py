from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer
import yaml
import Trust
import pandas as pd
import dill
from pathlib import Path

pd.reset_option("max_columns")
pd.set_option('display.max_columns', None)
file = 'german.data'

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
pd_dataset = pd.read_csv(file, names=header, delimiter= ' ')
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
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.01, random_state=1)

# TRAIN A RANDOM FOREST CLASSIFIER AND CREATE A TABULAR EXPLAINER
rf_classifier = RandomForestClassifier(max_depth=10, n_estimators=100,criterion='gini')
rf = rf_classifier.fit(train_x, train_y)
lime_explainer = LimeTabularExplainer(train_x.values, feature_names=data_x.columns.values, class_names=[1,0])

### TRUST COMPUTATION AND CACHE MANAGEMENT ###
trust = None
try:
    cached_trust_file_path = Path('trust_cache')
    if cached_trust_file_path.is_file():
        with open(cached_trust_file_path, 'rb') as cached_trust_file:
            trust = dill.load(cached_trust_file)
            print("INFO: Loaded trust from cache file " + repr(cached_trust_file_path.absolute()))
except:
    pass
if trust == None:
    # LOAD YAML CONFIG FILE
    print("INFO: Trust cache not found in " + repr(cached_trust_file_path.absolute()))
    config_file = open('trust4RF_config.yml', mode='r')
    yaml_config = yaml.safe_load(config_file)
    config_file.close()

    print("Computing Trust...")
    trust = Trust.Trust(yaml_config, rf, test_x, test_y, lime_explainer)
    with open(cached_trust_file_path, 'wb') as cached_trust_file:
        dill.dump(trust, cached_trust_file)

trust_weighted_avrg = trust.compute_trust_weighted_average()
trust_bn = trust.compute_trust_bn()

print("Trust: Weighted Average (without feedback): " + repr(trust_weighted_avrg.score_trust_without_feedback))
print("Trust: Weighted Average (with feedback): " + repr(trust_weighted_avrg.score_trust_with_feedback))
print("Trust: Bayesian network (with feedback): " + repr(trust_bn.bn_assessment))