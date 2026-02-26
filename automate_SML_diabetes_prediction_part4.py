import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df_raw):
    """
    Performs automated preprocessing steps on the diabetes dataset.
    Includes handling outliers (capping using IQR) and scaling numerical features.

    Args:
        df_raw (pd.DataFrame): The raw DataFrame containing the diabetes prediction data.

    Returns:
        pd.DataFrame: The preprocessed DataFrame, ready for machine learning tasks.
    """
    df_processed = df_raw.copy()

    numerical_cols_for_processing = df_processed.select_dtypes(include=np.number).columns.tolist()
    if 'Outcome' in numerical_cols_for_processing:
        numerical_cols_for_processing.remove('Outcome')

    for col in numerical_cols_for_processing:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_processed[col] = np.where(df_processed[col] < lower_bound, lower_bound, df_processed[col])
        df_processed[col] = np.where(df_processed[col] > upper_bound, upper_bound, df_processed[col])

    scaler = StandardScaler()

    df_processed[numerical_cols_for_processing] = scaler.fit_transform(df_processed[numerical_cols_for_processing])

    return df_processed
