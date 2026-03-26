import pandas as pd

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

# separate features and target
X = df.drop("Sleep Disorder", axis=1)
y = df["Sleep Disorder"]

print("\nTarget counts:")
print(y.value_counts())