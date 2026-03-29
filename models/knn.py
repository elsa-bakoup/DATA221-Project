import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
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
def build_knn_model(X_train, X_test, y_train, y_test):
    knn_model=Pipeline([
        ("preprocess_data", build_processor(X_train)),
        ("model", KNeighborsClassifier())
    ])
    knn_model.fit(X_train, y_train)
    predicted_result = knn_model.predict(X_test)
    return predicted_result, y_test

