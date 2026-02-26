
import pandas as pd
import os
import sys

# Add the current directory to sys.path to import the preprocessing module
# Assuming the runner script is in the same directory as automate_SML_diabetes_prediction_part4.py
current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_script_dir)

from automate_SML_diabetes_prediction_part4 import preprocess_data

# Define paths relative to the runner script location
raw_data_name = 'diabetes.csv'
processed_data_name = 'processed_diabetes.csv'
raw_data_path = os.path.join(current_script_dir, raw_data_name)
processed_data_path = os.path.join(current_script_dir, processed_data_name)

print(f"Attempting to load raw data from: {raw_data_path}")
if os.path.exists(raw_data_path):
    df_raw = pd.read_csv(raw_data_path)
    print("Raw data loaded successfully.")

    print("Starting data preprocessing...")
    df_processed = preprocess_data(df_raw)
    print("Data preprocessing complete.")

    print(f"Saving processed data to: {processed_data_path}")
    df_processed.to_csv(processed_data_path, index=False)
    print("Processed data saved successfully.")
else:
    print(f"Error: Raw data file '{raw_data_name}' not found at '{raw_data_path}'.")
