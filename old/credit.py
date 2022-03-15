from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,roc_auc_score, classification_report
from sklego.metrics import p_percent_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.explainers import MetricTextExplainer
from aif360.datasets import StandardDataset
from aif360.sklearn.metrics import disparate_impact_ratio
from lime.lime_tabular import LimeTabularExplainer
from aix360.metrics import faithfulness_metric, monotonicity_metric
from art.metrics import RobustnessVerificationTreeModelsCliqueMethod
from art.estimators.classification import SklearnClassifier
import numpy as np

import pandas as pd

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
print(pd_dataset.shape)
print(pd_dataset.columns)
pd_dataset.head(10)

# CLASSIFICATION COLUMN. [1,2] -> [1,0]
pd_dataset.classification.replace([1,2], [1,0], inplace=True)
pd_dataset.classification.value_counts()

# AIF360
# aifds = StandardDataset(pd_dataset, label_name='classification', favorable_classes=[1], protected_attribute_names=['statussex'],
#  privileged_classes=[['A91', 'A93', 'A94']])

# text_expl = MetricTextExplainer(aifds)
# print(text_expl.disparate_impact)

# LABEL ENCODING FOR CATEGORICAL/ORDINAL FEATURES
pd_dataset = pd.get_dummies(pd_dataset)
print(pd_dataset.columns)

# TRAIN-TEST SPLIT
# SPLIT FEATURES-TARGET (X-Y)
data_x = pd_dataset.drop(label_column, axis=1)
data_y = pd_dataset[label_column]
# SPLIT DATA
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.1, random_state=1)

# TRAIN A RANDOM FOREST CLASSIFIER
print('### PERFORMANCE (ACCURACY, RECALL, F1-SCORE ON THE TEST SET ###')
rf_classifier = RandomForestClassifier(max_depth=10, n_estimators=100,criterion='gini')
rf = rf_classifier.fit(train_x, train_y)

# TRAINING SET DISPARATE IMPACT RATIO
# train_dis = disparate_impact_ratio(train_y)
# print(f'Training set disparate impact: {train_dis:.3f}')

# USE IT TO PREDICT THE TEST SET
prediction = rf.predict(test_x)

# PRINT METRICS
print(classification_report(test_y, prediction))
print(confusion_matrix(test_y, prediction))

### FAIRNESS ###
# MODEL P-PERCENTAGE SCORE (On the test set)
print('### FAIRNESS (P-PERCENTAGE ON PROTECTED ATTRIBUTE STATUSSEX OF THE TEST SET ###')
print('p_percent_score A91 (MALE divorced/separated):', p_percent_score(sensitive_column="statussex_A91")(rf, test_x))
print('p_percent_score A92 (FEMALE divorced/separated/married): ', p_percent_score(sensitive_column="statussex_A92")(rf, test_x))
print('p_percent_score A93 (MALE single):', p_percent_score(sensitive_column="statussex_A93")(rf, test_x))
print('p_percent_score A94 (MALE married/widowed):', p_percent_score(sensitive_column="statussex_A94")(rf, test_x))

### ROBUSTNESS ###
# Clique Method Robustness Verification (on the test set)
rf_skmodel = SklearnClassifier(model=rf)
rt = RobustnessVerificationTreeModelsCliqueMethod(classifier=rf_skmodel)
average_bound, verified_error = rt.verify(x=test_x.values, y=pd.get_dummies(test_y).values, eps_init=0.3)#, eps_init=0.3, nb_search_steps=10, max_clique=2, 
                                         # max_level=2)

print('Average bound:', average_bound)
print('Verified error at eps:', verified_error)

### EXPLAINABILITY ###
# LIME Explanator
# Lime requires NP arrays, pandas dfs are not compatible, so convert them with .values
print('### EXPLAINABILITY (FAITHFULNESS AND MONOTONICITY METRICS OF THE LIME EXPLAINER ON THE TEST SET ###')
lime_explainer = LimeTabularExplainer(train_x.values, feature_names=data_x.columns.values, class_names=[1,0])
# Compute faithfulness and monotonicity metrics from the LIME explanations
ncases = test_x.values.shape[0]
print(ncases)
mon = np.zeros(ncases)
fait = np.zeros(ncases)
for i in range(ncases):
    predicted_class = rf.predict(test_x.values[i].reshape(1,-1))[0]
    exp = lime_explainer.explain_instance(test_x.values[i], rf.predict_proba, num_features=5, top_labels=1)
    le = exp.local_exp[predicted_class]
    m = exp.as_map()
    
    x = test_x.values[i]
    coefs = np.zeros(x.shape[0])
    
    for v in le:
        coefs[v[0]] = v[1]
    base = np.zeros(x.shape[0])

    mon[i] = monotonicity_metric(rf, test_x.values[i], coefs, base)
    fait[i] = faithfulness_metric(rf, test_x.values[i], coefs, base)
print("% of test records where explanation is monotonic",np.mean(mon))
print("Faithfulness metric mean: ",np.mean(fait))
print("Faithfulness metric std. dev.:", np.std(fait))


