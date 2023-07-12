from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from trustML.computation import TrustComputation
# NOTE: THIS DEMO REQUIRES THE XLRD PACKAGE. YOU CAN INSTALL IT THROUGH "pip install xlrd"

pd.reset_option("max_columns")
pd.set_option('display.max_columns', None)

demo_path = 'demos/credit_card_default/'
file_dataset = demo_path + 'credit_card_default.xls'
path_configuration_bank = demo_path + 'config_credit_card_default.yml'


# Load the data
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
    Y, test_size = 0.2)

# TRAIN A RANDOM FOREST CLASSIFIER
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, Y_train)

# TRUST STUFF
trust_bank = TrustComputation()
trust_bank.load_trust_definition(config_path=path_configuration_bank)
trust_bank.compute_trust(trained_model=rf_classifier, data_x=X_test, data_y=Y_test)

print(trust_bank.get_trust_as_JSON())

trust_bank.generate_trust_PDF(demo_path + "report_credit_card_default.pdf")