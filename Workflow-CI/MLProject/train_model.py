
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import sys

# Set MLflow experiment name
mlflow.set_experiment("Diabetes Prediction Model Training CI/CD")

# Enable MLflow autologging
mlflow.autolog()

# Define paths relative to the script's execution environment
# Assuming train_model.py is in 'MLProject' (which is inside 'Workflow-CI')
# and processed_diabetes.csv is in 'SML-diabetes_prediction_part4'
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up two directories to reach the base_path, then down to SML-diabetes_prediction_part4
processed_data_path = os.path.join(script_dir, '..', '..', 'SML-diabetes_prediction_part4', 'processed_diabetes.csv')

print(f"Loading preprocessed data from: {processed_data_path}")
if os.path.exists(processed_data_path):
    df = pd.read_csv(processed_data_path)
    print("Preprocessed data loaded successfully.")
else:
    print(f"Error: Processed data file not found at {processed_data_path}")
    sys.exit(1)

# Separate features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data split into training and testing sets.")

# Train a Logistic Regression model
with mlflow.start_run():
    model = LogisticRegression(random_state=42, solver='liblinear') # Using 'liblinear' solver for small datasets
    model.fit(X_train, y_train)
    print("Logistic Regression model trained successfully with MLflow autologging.")

# Optional: Make predictions and print a simple evaluation (MLflow autologs more comprehensive metrics)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy on test set: {accuracy:.4f}")
