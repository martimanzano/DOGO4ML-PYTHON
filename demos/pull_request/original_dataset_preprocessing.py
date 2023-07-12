from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import pandas as pd

file_dataset = 'demos/pull_request/new_pullreq.csv' # 2.2 GB DATASET FROM https://github.com/zhangxunhui/new_pullreq_msr2020

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

dataset.to_csv("new_pullreq-red.csv", sep=',', index=False)  # 19 MB FILE