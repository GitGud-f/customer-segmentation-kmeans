"""
Module: Data Loader
Description: 
    Handles the loading, inspection, cleaning, and preprocessing of the 
    Mall Customers dataset. It serves as the data access layer for the project.

Functions:
    - load_data: Reads CSV into a Pandas DataFrame.
    - inspect_data: Prints shape, info, and missing value statistics.
    - preprocess_data: Cleans data (drops columns) for analysis.
    - save_data: Persists processed data to disk.
"""

import pandas as pd
import os

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the CSV data from the specified filepath.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded data.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Data successfully loaded from {filepath}")
    return df


def inspect_data(df: pd.DataFrame):
    """
    Performs initial data exploration tasks: shape, dtypes, missing values, duplicates.
    Prints the results to stdout.
    
    Args:
        df (pd.DataFrame): Dataframe to inspect.
    """
    print("--- Data Inspection ---")
    print(f"Shape: {df.shape} (Rows, Columns)")
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicates}")
    
    if duplicates > 0:
        print("Warning: Duplicate values detected.")
    else:
        print("No duplicates found.")
        
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the data by dropping unnecessary columns for clustering.
    
    Args:
        df (pd.DataFrame): Raw dataframe.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df_clean = df.copy()
    
    if 'CustomerID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['CustomerID'])
        df_clean = df_clean.rename(columns={'pruchase spending': 'purchase spending'})
        print("Dropped 'CustomerID' column.")
    else:
        print("'CustomerID' column not found (already dropped?).")
        
    return df_clean

def save_data(df: pd.DataFrame, filepath: str):
    """
    Saves the dataframe to a CSV file.
    Ensures the directory exists before saving.
    
    Args:
        df (pd.DataFrame): Data to save.
        filepath (str): Destination path.
    """
    
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
        
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")