from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score, \
    classification_report
from sklearn.preprocessing import label_binarize

from models.logistic_regression import MODEL_NAME, tune_model
from shared.preprocess import build_processor
from shared.preprocess import clean_raw_data
# Model name
MODEL_NAME="KNN"
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
        ("preprocessor", build_processor(X_train)),
        ("classifier", KNeighborsClassifier())
    ])
    return knn_model


# Hyperparameter grid for tuning
param_grid_knn={
    "classifier__n_neighbors": [3,5,7,9,11],
    "classifier__weights": ["uniform","distance"],
    "classifier__metric": ["euclidean", "manhattan"]
}

def grid_search_knn(param_grid_knn, knn_model, X_train, y_train):
    cross_validation=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search_knn=GridSearchCV(
        knn_model,
        param_grid=param_grid_knn,
        cv=cross_validation,
        scoring="recall_macro",
    )
    grid_search_knn.fit(X_train, y_train)
    return grid_search_knn

def model_evaluation(grid_search_knn, X_test, y_test):
    y_predicted=grid_search_knn.predict(X_test)
    y_predicted_prob=grid_search_knn.predict_proba(X_test)

    categories=sorted(y_test.unique())
    y_test_binary=label_binarize(y_test,categories=categories)

    confusionMatrix=confusion_matrix(y_test, y_predicted)

    metrics={
        'model': MODEL_NAME,
        'Accuracy': accuracy_score(y_test, y_predicted),
        'Precision': precision_score(y_test, y_predicted, average='macro'),
        'Recall': recall_score(y_test, y_predicted, average='macro'),
        'f1-score': f1_score(y_test, y_predicted, average='macro'),
        'roc_auc': roc_auc_score(y_test_binary, y_predicted_prob, average='macro')
    }

    metrics_report=classification_report(y_test, y_predicted)
    return metrics, metrics_report, confusionMatrix, categories

# Export & save confusion matrix of KNN
def save_confusion_matrix(confusionMatrix, categories):
    fig, ax=plt.subplots(figsize=(6,6))
    displayMatrix=confusion_matrix(confusion_matrix=confusionMatrix, labels=categories)
    displayMatrix.plot(ax=ax)
    plt.title("KNN Confusion Matrix")
    plt.savefig("../results/figures/knn_confusion_matrix.png")
    plt.close(fig)

def save_best_param(best_param):
    df=pd.DataFrame([best_param])
    path=Path("../results/tuning/knn_best_param.csv")
    df.to_csv(path, index=False)

def save_metrics(metrics):
    df=pd.DataFrame([metrics])
    path=Path("../results/metrics/knn_metrics.csv")
    df.to_csv(path, index=False)


def main():
    X_train, X_test, y_train, y_test=split_data(load_clean_data())

    preprocessor=build_processor(X_train)
    knn_model=build_knn_model(preprocessor)
    grid=tune_model(knn_model, param_grid_knn, X_train, y_train)

    best_model=grid.best_estimator_
    best_param=grid.best_params_

    metrics, metrics_report, confusionMatrix, categories=model_evaluation(grid, X_test, y_test)

    save_best_param(best_param)
    save_metrics(metrics)
    save_confusion_matrix(confusionMatrix, categories)
    save_permutation_importance(best_model)



