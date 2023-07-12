from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from dill import dump
from lime.lime_tabular import LimeTabularExplainer
from trustML.computation import TrustComputation

pd.reset_option("max_columns")
pd.set_option('display.max_columns', None)

demo_path = 'demos/pull_request/'
path_configuration = demo_path + 'config_pull_request.yml'
file_dataset = demo_path + 'new_pullreq-red.csv'

# Load the data
dataset = pd.read_csv(file_dataset, sep=",", header=0)
dataset.head(0)

# Extract target column
Y = dataset["merged_or_not"]
Y.describe()

# Decompose the dataset: Training and test split and drop target
X_train, X_test, Y_train, Y_test = train_test_split(dataset.drop(columns=["merged_or_not"]), Y, test_size=0.2, stratify=dataset.merged_or_not, random_state=1)

# TRAIN A RANDOM FOREST CLASSIFIER
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, Y_train)

# TRAIN A LIME TABULAR EXPLAINER
lime_explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns.values, class_names=[1,0])

# SAVE EXPLAINER TO THE DEMO DIRECTORY (OPTIONAL, ONLY NECESSARY IF THEY ARE NOT ALREADY PRESENT)
with open(demo_path + 'lime_explainer', 'wb') as explainer_file:
    dump(lime_explainer, explainer_file)

# TRUST STUFF
trust_pr = TrustComputation()
trust_pr.load_trust_definition(config_path=path_configuration)
trust_pr.compute_trust(trained_model=rf_classifier, data_x=X_test, data_y=Y_test)

print(trust_pr.get_trust_as_JSON())

trust_pr.generate_trust_PDF(demo_path + "reportPR.pdf")