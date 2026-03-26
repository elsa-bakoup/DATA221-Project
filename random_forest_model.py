import pandas as pd
from sklearn.preprocessing import LabelEncoder

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