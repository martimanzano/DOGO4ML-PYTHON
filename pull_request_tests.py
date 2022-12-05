from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils import resample
from trust.Trust import Trust, Evaluation_method
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


pd.reset_option("max_columns")
pd.set_option('display.max_columns', None)

file_dataset = 'C:/Users/marti/Desktop/new_pullreq.csv'

yaml_configuration_to_use = 'configs/trust4RF_config_retention_corporate.yml'

# Load the data
dataset = pd.read_csv(file_dataset, sep=",", header=0)
dataset.head(0)

# Interchange target to compute F1 with the positive class as minority class
#dataset['merged_or_not'].replace([0,1], [1,0], inplace=True)
dataset.merged_or_not.value_counts()

# do some preprocessing and cleaning
columns_to_drop = ["id", "project_id", "ownername", "reponame", "mergetime_minutes", "bug_fix", "ci_latency", "first_response_time", "inte_affiliation", "inte_country", "contrib_affiliation"]

dataset.drop(columns=columns_to_drop, inplace=True)
dataset[["same_country", "same_affiliation"]] = dataset[["same_country", "same_affiliation"]].fillna(0)
dataset[["contrib_country", "ci_first_build_status", "ci_last_build_status"]] = dataset[["contrib_country", "ci_first_build_status", "ci_last_build_status"]].fillna("unknown") # ci_first_build_status, ci_last_build_status

# Remove NAs
dataset = dataset.dropna()

# Transform categorical columns using label encoder
categorical_columns = ["ci_first_build_status", "ci_last_build_status", "language", "contrib_gender", "contrib_country", "contrib_first_emo", "inte_first_emo"]

dataset[categorical_columns] = dataset[categorical_columns].apply(LabelEncoder().fit_transform)

dataset.dtypes.to_list()

### DOWN SAMPLING ###
# Separate majority and minority classes
df_majority = dataset[dataset.merged_or_not==1]
df_minority = dataset[dataset.merged_or_not==0]
len(df_minority)

# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,                  # sample without replacement
                                 n_samples=len(df_minority),     # to match minority class
                                 random_state=123)               # reproducible results
 
# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
 
# Display old and new class counts
dataset.merged_or_not.value_counts()
df_downsampled.merged_or_not.value_counts()

dataset = df_downsampled

# Extract target column
Y = dataset["merged_or_not"]
Y.describe()

# Decompose the dataset: Training and test split and drop target
X_train, X_test, Y_train, Y_test = train_test_split(dataset.drop(columns=["merged_or_not"]), Y, test_size=0.3, stratify=dataset.merged_or_not, random_state=1)

# TRAIN A RANDOM FOREST CLASSIFIER
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, Y_train)

# Perform basic accuracy tests
prediction = rf_classifier.predict(X_test)
perf_dict = classification_report(Y_test, prediction, output_dict=True)

perf_accuracy = perf_dict['accuracy']
perf_accuracy
perf_precision = perf_dict['weighted avg']['precision']
perf_recall = perf_dict['weighted avg']['recall']
perf_f1 = perf_dict['weighted avg']['f1-score']
perf_dict

roc_auc = roc_auc_score(Y_test, prediction)
roc_auc

# Plot features' imporance
importances = rf_classifier.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_classifier.estimators_], axis=0)

forest_importances = pd.Series(importances, index=dataset.drop(columns=["merged_or_not"]).columns)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
fig.show()




