from os import path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
from trust.Trustable_entity import TrustableEntity
from trust.assessment_methods.Trust_WA import Trust_Weighted_Average
from trust.assessment_methods.Trust_BN import Trust_BN

pd.reset_option("max_columns")
pd.set_option('display.max_columns', None)
file_dataset = 'german.data'
path_configuration_to_use = 'configs/newTrust_config_german_credit_banker.yml'

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
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.3, random_state=2)

# TRAIN A RANDOM FOREST CLASSIFIER AND CREATE A TABULAR EXPLAINER
rf_classifier = RandomForestClassifier(max_depth=10, n_estimators=100,criterion='gini')
rf = rf_classifier.fit(train_x, train_y)
lime_explainer = LimeTabularExplainer(train_x.values, feature_names=data_x.columns.values, class_names=[1,0])

# TRUST STUFF
cache_path = 'cache_new_structure'

if path.isfile(cache_path):
    # IF CACHE FILE IS PRESENT, LOAD IT
    entity = TrustableEntity.load_cache(cache_path)
else:
    # IF NOT, WE INSTANTIATE A TRUSTABLE ENTITY, COMPUTE ITS METRICS AND SAVE IT TO A CACHE FILE
    entity = TrustableEntity(config_path=path_configuration_to_use, trained_model=rf, data_x=test_x, data_y=test_y, data_E = None, explainer = lime_explainer)
    entity.assess_trust_metrics()
    entity.save_cache(cache_path)

# TRUST WITH WA, WE INSTANTIATE THE TRUST WA, ASSESS IT AND PRINT THE RESULT
# entity.config = entity.load_config(path_configuration_to_use) # RELOAD WA HIERARCHY AND WEIGHTS IF THERE'VE BEEN CHANGES SINCE WE COMPUTED THE METRICS
trust_wa = Trust_Weighted_Average(trustable_entity=entity)
trust_wa.assess()
trust_wa.print_weighted_scores_tree() # PRETTY PRINT THE WA ASSESSMENT
print(trust_wa.get_weighted_scores_tree_as_JSON()) # GET WA ASSESSMENT AS JSON

# TRUST WITH BN
#trust_bn = Trust_BN(trustable_entity=entity)
#trust_bn.assess()
#print(trust_bn.get_bn_assessment())