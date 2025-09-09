import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("titanic_dataset.csv")
print("Original Dataset Shape:",df.shape)
print(df.head())

cols_to_drop = ['Name','Ticket','Cabin','PassengerId']
df = df.drop(cols_to_drop, axis=1, errors="ignore")
print("\nAfter Dropping Unwanted Columns:", df.shape)

df['Sex'] = LabelEncoder().fit_transform(df['Sex'])  # Female=0, Male=1
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print("\nAfter Encoding:")
print(df.head())

numeric_cols = ['Age', 'Fare']
num_data = df[numeric_cols].copy()

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
    return series / (10 ** j)

df_decimal = num_data.apply(decimal_scaling)


for feature in numeric_cols:
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.hist(num_data[feature], bins=20, color='skyblue', edgecolor='black')
    plt.title(f"Original {feature}")

    plt.subplot(2, 2, 2)
    plt.hist(df_minmax[feature], bins=20, color='lightgreen', edgecolor='black')
    plt.title("Min-Max Normalization")

    plt.subplot(2, 2, 3)
    plt.hist(df_zscore[feature], bins=20, color='orange', edgecolor='black')
    plt.title("Z-Score Standardization")

    plt.subplot(2, 2, 4)
    plt.hist(df_decimal[feature], bins=20, color='violet', edgecolor='black')
    plt.title("Decimal Scaling")

    plt.suptitle(f"Normalization Comparison for '{feature}'", fontsize=14)
    plt.tight_layout()
    plt.show()


X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\nModel Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
