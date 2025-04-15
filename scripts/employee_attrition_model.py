# Predicting Attrition with Logistic Regression
# employee_attrition_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the dataset
url = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
df = pd.read_csv(url)
print("Initial shape:", df.shape)

# Step 2: Clean the dataset
df.drop(columns=["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"], inplace=True)
df["Attrition"] = df["Attrition"].apply(lambda x: 1 if x == "Yes" else 0)

# Step 3: Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)
print("New shape after encoding:", df_encoded.shape)

# Step 4: Train-test split
X = df_encoded.drop("Attrition", axis=1)
y = df_encoded["Attrition"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train logistic regression
# Logistic regression with solver fix
model = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Balanced Model Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = rf_model.predict(X_test)

print("\n--- Random Forest Performance ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Step 5a: Feature interpretation
import numpy as np

# Get feature names and coefficients
feature_names = X.columns
coefficients = model.coef_[0]

# Combine and sort
coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

# Sort by impact
coef_df_sorted = coef_df.sort_values(by="Coefficient", ascending=False)

# Display top positive and negative predictors
print("\nTop Features Pushing Toward Attrition (Positive Coefficients):")
print(coef_df_sorted.head(10))

print("\nTop Features Pushing Toward Staying (Negative Coefficients):")
print(coef_df_sorted.tail(10))

# Step 6: Evaluate model
y_pred = model.predict(X_test)

print("\n--- Model Performance ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature importance
importances = rf_model.feature_importances_
feat_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Important Features (Random Forest):")
print(feat_df.head(10))
