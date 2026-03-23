import pandas as pd
import zipfile

with zipfile.ZipFile("sleep-health-and-lifestyle-dataset.zip", "r") as zip_ref:
    zip_ref.extractall("data")

df = pd.read_csv("data/Sleep_health_and_lifestyle_dataset.csv")
print(df.head())