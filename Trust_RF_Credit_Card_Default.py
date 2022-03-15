from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer
from trust.Trust import Trust, Evaluation_method
import pandas as pd

pd.reset_option("max_columns")
pd.set_option('display.max_columns', None)

yaml_configuration_corp = 'configs/trust4RF_config_credit_card_default_corporate.yml'
yaml_configuration_banker = 'configs/trust4RF_config_credit_card_default_banker.yml'

yaml_configuration_to_use = yaml_configuration_banker

# Load the data
file_dataset = 'credit_card_default.xls'
dataset = pd.read_excel(file_dataset, header=1).drop(columns=['ID']).rename(columns={'PAY_0':'PAY_1'})
dataset.head()

# Extract the target
Y = dataset["default payment next month"]
categorical_features = ['EDUCATION', 'MARRIAGE','PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
for col in categorical_features:
    dataset[col] = dataset[col].astype('category')

# SENSITIVE COLUMN [2 (female), 1 (male)] -> [0,1]
dataset['SEX'].replace([2,1], [0,1], inplace=True)

# Decompose the dataset
# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(dataset.drop(columns=['default payment next month']), 
    Y, test_size = 0.001, random_state=1)

# TRAIN A RANDOM FOREST CLASSIFIER
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, Y_train)

lime_explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns.values, class_names=[1,0])

### TRUST COMPUTATION AND CACHE MANAGEMENT ###
trust = Trust.load_compute_trust_with_cache(file_dataset, yaml_configuration_to_use, X_test, Y_test, rf_classifier, lime_explainer=lime_explainer)
trust_weighted_avrg = trust.evaluate_trust(yaml_configuration_to_use, Evaluation_method.Weighted_Average)

print(trust.get_Trust_metrics_as_JSON())
print(trust_weighted_avrg)
print(trust_weighted_avrg.get_Trust_WA_as_JSON())
#trust_bn = trust.evaluate_trust(yaml_configuration_to_use, Evaluation_method.Bayesian_Network)
# print("Trust: Bayesian network: " + repr(trust_bn.bn_assessment))