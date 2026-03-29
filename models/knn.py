from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import zipfile
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

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

accuracyList=[]

for train_index, test_index in skf.split(features_train, labels_train):
    X_train, X_test = features_train.iloc[train_index], features_train.iloc[test_index]
    y_train, y_test = labels_train.iloc[train_index], labels_train.iloc[test_index]

    model=KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    predicted_result = model.predict(X_test)
    accuracyList.append(accuracy_score(y_test, predicted_result))

# Train model using full set of training data
final_model=KNeighborsClassifier(n_neighbors=5)
final_model.fit(features_train, labels_train)

# Predict values
predicted_result =final_model.predict(features_test)

# Confusion matrix
confusionMatrix=confusion_matrix(labels_test, predicted_result)
labels=["None", "Insomnia", "Sleep Disorder"]
confusionMatrix_df=pd.DataFrame(
    confusionMatrix,
    index=labels,
    columns=labels)

# Accuracy
accuracy=sum(accuracyList)/len(accuracyList)

# Precision
precision=precision_score(labels_test, predicted_result, average='weighted')

# recall
recall=recall_score(labels_test, predicted_result, average='weighted')

# f1score
f1=f1_score(labels_test, predicted_result, average='weighted')

# Display result
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)
print("Confusion Matrix: \n", confusionMatrix_df)
