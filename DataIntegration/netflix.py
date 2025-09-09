# netflix.py - Data Cleaning, Integration & Normalization (DPV Project)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ------------------------------------------------------------
# STEP 1: Load Dataset (robust loading)
# ------------------------------------------------------------
file_name = "netflix_titles.csv"

# Handle Kaggle-style nested folder issue
if not os.path.exists(file_name):
    nested_path = os.path.join("netflix_titles.csv", "netflix_titles.csv")
    if os.path.exists(nested_path):
        file_name = nested_path

# Try multiple encodings
try:
    df = pd.read_csv(file_name, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(file_name, encoding="ISO-8859-1")

print("✅ Dataset Loaded Successfully")
print("Shape:", df.shape)

# Drop extra "Unnamed" columns if present
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("\nColumns after cleanup:\n", df.columns.tolist())
print("\nFirst 5 Records:\n", df.head())

# ------------------------------------------------------------
# STEP 2: Summary & Missing Values
# ------------------------------------------------------------
print("\nSummary Report (Numerical):\n", df.describe(include=[np.number]))
print("\nMissing Values:\n", df.isnull().sum())

# Handle missing values
df["director"] = df["director"].fillna("unknown")
df["cast"] = df["cast"].fillna("unknown")
df["country"] = df["country"].fillna("unknown")
df["date_added"] = df["date_added"].fillna(df["date_added"].mode()[0])
df["rating"] = df["rating"].fillna("unknown")
df["duration"] = df["duration"].fillna("unknown")

# ------------------------------------------------------------
# STEP 3: Duplicates
# ------------------------------------------------------------
dup_count = df.duplicated().sum()
print("\nDuplicate Rows Found:", dup_count)
df = df.drop_duplicates()
print("After Removing Duplicates:", df.shape)

# ------------------------------------------------------------
# STEP 4: Text Cleaning
# ------------------------------------------------------------
text_cols = ["title", "director", "cast", "country", "listed_in", "description"]
for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()

# ------------------------------------------------------------
# STEP 5: Outlier Detection (IQR on release_year)
# ------------------------------------------------------------
Q1 = df["release_year"].quantile(0.25)
Q3 = df["release_year"].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df["release_year"] < (Q1 - 1.5*IQR)) | (df["release_year"] > (Q3 + 1.5*IQR))]
print("\nOutliers in release_year:", outliers.shape[0])
df = df.drop(outliers.index)

# Save cleaned dataset
df.to_csv("netflix_cleaned.csv", index=False)
print("✅ Cleaned dataset saved as netflix_cleaned.csv")

# ------------------------------------------------------------
# STEP 6: Normalization (Movies Only)
# ------------------------------------------------------------
# Work only with Movies (ignore TV Shows for duration)
df_movies = df[df["type"] == "movie"].copy()

# Filter only rows with numeric duration (ending in "min")
df_movies = df_movies[df_movies["duration"].str.contains("min", na=False)]

# Convert duration to float
df_movies["duration"] = df_movies["duration"].str.replace(" min", "", regex=False).astype(float)

numeric_cols = ["release_year", "duration"]
num_data = df_movies[numeric_cols].dropna()

print("\nNumeric Data Shape for Normalization:", num_data.shape)

if not num_data.empty:
    # Min-Max Normalization
    scaler_minmax = MinMaxScaler()
    df_minmax = pd.DataFrame(scaler_minmax.fit_transform(num_data), columns=numeric_cols)

    # Z-Score Standardization
    scaler_zscore = StandardScaler()
    df_zscore = pd.DataFrame(scaler_zscore.fit_transform(num_data), columns=numeric_cols)

    # Decimal Scaling
    def decimal_scaling(series):
        max_val = series.abs().max()
        j = len(str(int(max_val)))
        return series / (10**j)

    df_decimal = num_data.apply(decimal_scaling)

    # Compare sample
    sample = pd.concat([
        num_data.head(),
        df_minmax.head(),
        df_zscore.head(),
        df_decimal.head()
    ], axis=1)
    print("\nComparison Table (Original | MinMax | Zscore | Decimal):\n", sample)

    # Visualization
    feature = "duration"
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.hist(num_data[feature], bins=20, color="skyblue", edgecolor="black")
    plt.title("Original")

    plt.subplot(1, 3, 2)
    plt.hist(df_minmax[feature], bins=20, color="lightgreen", edgecolor="black")
    plt.title("Min-Max")

    plt.subplot(1, 3, 3)
    plt.hist(df_zscore[feature], bins=20, color="orange", edgecolor="black")
    plt.title("Z-Score")

    plt.suptitle(f"Normalization Comparison for {feature}")
    plt.tight_layout()
    plt.show()

    # Save standardized dataset
    df_zscore.to_csv("netflix_standardized.csv", index=False)
    print("✅ Standardized dataset saved as netflix_standardized.csv")

else:
    print("\n⚠️ No numeric movie durations found for normalization. Skipping normalization step.")
