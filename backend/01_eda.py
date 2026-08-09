import pandas as pd

# Load the dataset
df = pd.read_csv("dataset/dataset.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Shape of dataset
print("\nShape:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns.tolist())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Target distribution
print("\nTarget Distribution:")
print(df["Target"].value_counts())