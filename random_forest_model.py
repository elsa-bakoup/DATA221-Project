import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# load the dataset
df = pd.read_csv("data/raw/Sleep_health_and_lifestyle_dataset.csv")

# check the first rows
print(df.head())

# check column names
print(df.columns)

# fix target column
df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")

# remove id column
df = df.drop("Person ID", axis=1)

# split blood pressure into two numeric columns
df[["Systolic", "Diastolic"]] = df["Blood Pressure"].str.split("/", expand=True)

# convert the new blood pressure columns to integers
df["Systolic"] = df["Systolic"].astype(int)
df["Diastolic"] = df["Diastolic"].astype(int)

# drop the original blood pressure column
df = df.drop("Blood Pressure", axis=1)
# separate features and target
X = df.drop("Sleep Disorder", axis=1)
y = df["Sleep Disorder"]

print("\nTarget counts:")
print(y.value_counts())

# encode categorical feature columns
X = pd.get_dummies(X, drop_first=True)

print("\nEncoded feature columns:")
print(X.columns)

# encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

print("\nEncoded target counts:")
print(pd.Series(y).value_counts())

#train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#random forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)

#predictions + evaluation
y_pred = rf.predict(X_test)

print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))
print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))

#feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))