from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize

from shared.preprocess import build_processor
from shared.preprocess import clean_raw_data
from sklearn.metrics import ConfusionMatrixDisplay

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

    X_train, X_test, y_train, y_test=train_test_split(X, y,stratify=y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Build knn model
def build_knn_model(preprocessor):
    knn_model=Pipeline([
        ("preprocessor", preprocessor),
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
    grid_knn=GridSearchCV(
        knn_model,
        param_grid=param_grid_knn,
        cv=cross_validation,
        scoring="recall_macro",
    )
    grid_knn.fit(X_train, y_train)
    return grid_knn

def model_evaluation(grid_search_knn, X_test, y_test):
    y_predicted=grid_search_knn.predict(X_test)
    y_predicted_prob=grid_search_knn.predict_proba(X_test)

    classes=sorted(y_test.unique())
    y_test_binary=label_binarize(y_test,classes=classes)

    cm=confusion_matrix(y_test, y_predicted)

    metrics={
        'model': MODEL_NAME,
        'Accuracy': accuracy_score(y_test, y_predicted),
        'Precision': precision_score(y_test, y_predicted, average='macro'),
        'Recall': recall_score(y_test, y_predicted, average='macro'),
        'f1-score': f1_score(y_test, y_predicted, average='macro'),
        'roc_auc': roc_auc_score(y_test_binary, y_predicted_prob,multi_class='ovr', average='macro')
    }

    metrics_report=classification_report(y_test, y_predicted)
    return metrics, metrics_report, cm, classes

# Export & save confusion matrix of KNN
def save_confusion_matrix(confusionMatrix, classes):
    fig, ax=plt.subplots(figsize=(6,6))
    displayMatrix=ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=classes)
    displayMatrix.plot(ax=ax, colorbar=False)
    plt.title("KNN Confusion Matrix")
    plt.tight_layout()
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

def save_permutation_importance(best_model, X_test, y_test):
    importance=permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=30,
        random_state=42,
        scoring="recall_macro"
    )

    importance_df=pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std
     }).sort_values(by="importance_mean", ascending=False)

    importance_df.to_csv("../results/interpretability/knn/knn_permutation_importance.csv", index=False)

    plt.figure(figsize=(10,6))
    plt.barh(importance_df["feature"],
             importance_df["importance_mean"],
             xerr=importance_df["importance_std"])
    plt.title("KNN Permutation Importance")
    plt.xlabel("Permutation Importance (mean decrease in accuracy)")
    plt.ylabel("Feature")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("../results/interpretability/knn/knn_permutation_importance.png", dpi=300)
    plt.close()


def main():
    X_train, X_test, y_train, y_test=split_data(load_clean_data())

    preprocessor=build_processor(X_train)
    knn_model=build_knn_model(preprocessor)
    grid=grid_search_knn(param_grid_knn, knn_model, X_train, y_train)

    best_model=grid.best_estimator_
    best_param=grid.best_params_

    metrics, metrics_report, confusionMatrix, classes=model_evaluation(grid, X_test, y_test)

    save_best_param(best_param)
    save_metrics(metrics)
    save_confusion_matrix(confusionMatrix, classes)
    save_permutation_importance(best_model,X_test,y_test)

    print("Best parameters: ", best_param)

    print("Final Test Metrics: ")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("Metrics Report: ")
    print(metrics_report)

main()


