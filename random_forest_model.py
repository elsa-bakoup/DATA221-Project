import pandas as pd

# load the dataset
df = pd.read_csv("data/raw/Sleep_health_and_lifestyle_dataset.csv")

# check the first rows
print(df.head())

# check column names
print(df.columns)