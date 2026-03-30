import json
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from shared.preprocess import build_processor

# Model name
MODEL_NAME = "decision_tree"

def load_split_data():
    X_train = pd.read_csv("../data/split/X_train.csv")
    X_test = pd.read_csv("../data/split/X_test.csv")
    y_train = pd.read_csv("../data/split/y_train.csv", keep_default_na=False).squeeze("columns")
    y_test = pd.read_csv("../data/split/y_test.csv", keep_default_na=False).squeeze("columns")

    return X_train, X_test, y_train, y_test


def build_model(preprocessor):
    decision_tree_pipeline_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ])

    return decision_tree_pipeline_model


# Hyperparameter grid (based on your original config + tuning)
param_grid = {
    "classifier__criterion": ["gini", "entropy"],
    "classifier__max_depth": [3, 5, 7, 10, None],
    "classifier__min_samples_split": [2, 4, 6, 10],
    "classifier__min_samples_leaf": [1, 2, 4]
}


def tune_model(model, param_grid, X_train, y_train):
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=cross_validation,
        scoring="recall_macro",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid


def evaluate_model(best_model, X_test, y_test):
    y_pred = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)

    classes = sorted(y_test.unique())
    y_test_binary = label_binarize(y_test, classes=classes)

    roc_auc = roc_auc_score(
        y_test_binary,
        y_pred_prob,
        multi_class="ovr",
        average="macro"
    )

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        'Model': MODEL_NAME,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average="macro", zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average="macro", zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average="macro", zero_division=0),
        'roc_auc': roc_auc,
    }

    report = classification_report(y_test, y_pred)

    return metrics, cm, classes, report


def save_best_params(best_params):
    with open("../results/tuning/decision_tree_best_params.json", "w", encoding="utf-8") as file:
        json.dump(best_params, file)


def save_metrics(metrics):
    pd.DataFrame([metrics]).to_csv(
        "../results/metrics/decision_tree_metrics.csv",
        index=False
    )


def save_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=False)
    plt.title("Decision Tree Confusion Matrix")
    plt.tight_layout()
    plt.savefig("../results/figures/decision_tree_confusion_matrix.png", dpi=300)
    plt.close(fig)


def save_feature_importance(best_model):
    preprocess = best_model.named_steps["preprocessor"]
    classifier = best_model.named_steps["classifier"]

    feature_names = preprocess.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(
        "../results/interpretability/decision_tree/feature_importance.csv",
        index=False
    )


def save_feature_importance_plot(best_model):
    preprocess = best_model.named_steps["preprocessor"]
    classifier = best_model.named_steps["classifier"]

    feature_names = preprocess.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance")

    plt.figure(figsize=(10, max(8, len(importance_df) * 0.3)))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.title("Decision Tree Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(
        "../results/interpretability/decision_tree/feature_importance_plot.png",
        dpi=300
    )
    plt.close()


def main():
    X_train, X_test, y_train, y_test = load_split_data()

    preprocessor = build_processor(X_train)
    model = build_model(preprocessor)

    grid = tune_model(model, param_grid, X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    metrics, cm, classes, report = evaluate_model(best_model, X_test, y_test)

    save_best_params(best_params)
    save_metrics(metrics)
    save_confusion_matrix(cm, classes)
    save_feature_importance(best_model)
    save_feature_importance_plot(best_model)

    print("Best Parameters:")
    print(best_params)

    print("\nFinal Test Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":
    main()