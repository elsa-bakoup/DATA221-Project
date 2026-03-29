import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score

from shared.preprocess import build_processor
from shared.preprocess import clean_raw_data

# Load dataset
dataset = pd.read_csv("data/Sleep_health_and_lifestyle_dataset.csv")

# Target variable
TARGET_COLUMN = "Sleep Disorder"

# Loading clean data
def load_clean_data():
    cleaned_data=clean_raw_data(dataset)
    return cleaned_data

def split_data(cleaned_data):
    X=cleaned_data.drop(columns=[TARGET_COLUMN])
    y=cleaned_data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Build knn model
def build_knn_model(X_train):
    knn_model=Pipeline([
        ("preprocess_data", build_processor(X_train)),
        ("model", KNeighborsClassifier())
    ])

    return knn_model

# Hyperparameter grid for tuning
param_grid_knn={
    "classifier__n_neighbors": [3,5,7,9,11],
    "classifier__weights": ["uniform","distance"],
    "classifier__metric": ["euclidean", "manhattan"]
}

X_train, X_test, y_train, y_test=split_data(load_clean_data())
knn_model=build_knn_model(X_train)

cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search_knn=GridSearchCV(
    knn_model,
    param_grid=param_grid_knn,
    cv=cv
)

# Train the model & prediction
knn_model.fit(X_train, y_train)
predicted_result=knn_model.predict(X_test)
predicted_prob=knn_model.predict_proba(X_test)

# Evaluation
accuracy=accuracy_score(y_test, predicted_result)
precision=precision_score(y_test, predicted_result, average='weighted')
recall=recall_score(y_test, predicted_result, average='weighted')
f1=f1_score(y_test, predicted_result, average='weighted')
roc=roc_auc_score(y_test, predicted_prob, multi_class='ovr', average='weighted')

# Confusion matrix
confusionMatrix_df=confusion_matrix(y_test, predicted_result)
labels=["None", "Insomnia", "Sleep Apnea"]
cm_df=pd.DataFrame(
    confusionMatrix_df,
    index=labels,
    columns=labels)

# Display result
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)
print("ROC AUC: ", roc)
print("Confusion Matrix: \n", cm_df)

