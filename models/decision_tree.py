from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import pandas as pd
import zipfile

# Loading the data from the zip file
with zipfile.ZipFile("sleep-health-and-lifestyle-dataset.zip", "r") as zip_ref:
    zip_ref.extractall("data")

dataset = pd.read_csv("data/Sleep_health_and_lifestyle_dataset.csv")

# Features + target variables
feature_matrix = dataset.drop(columns=['Sleep Disorder'])
feature_matrix = pd.get_dummies(feature_matrix)

target_variable = dataset["Sleep Disorder"].fillna("None")

# Train-test split
features_train, features_test, labels_train, labels_test = train_test_split(
    feature_matrix, target_variable, test_size=0.2, stratify=target_variable, random_state=42)

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracy_scores = []

for train_index, val_index in skf.split(features_train, labels_train):
    feature_tr = features_train.iloc[train_index]
    feature_val = features_train.iloc[val_index]
    labels_tr = labels_train.iloc[train_index]
    labels_val = labels_train.iloc[val_index]

    # new model each fold
    model = DecisionTreeClassifier(criterion='entropy', max_depth=7, min_samples_split=4)

    model.fit(feature_tr, labels_tr)
    predicted_val = model.predict(feature_val)

    accuracy_scores.append(accuracy_score(labels_val, predicted_val))

# Train final model on full training data
final_model = DecisionTreeClassifier(criterion='entropy', max_depth=7, min_samples_split=4)
final_model.fit(features_train, labels_train)

# Test predictions
predicted_labels = final_model.predict(features_test)
predicted_probabilities = final_model.predict_proba(features_test)