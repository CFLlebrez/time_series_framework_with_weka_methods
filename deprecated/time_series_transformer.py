#!/usr/bin/env python3
"""
Time Series Transformer for Multivariate Prediction Tasks

This script transforms a CSV file containing multivariate time series data into a format
suitable for regression analysis. It takes a time series where each row is a temporal record
with multiple variables and transforms it into a format where each row contains:
- Past values of all variables (for context)
- Future values of the target variable (for prediction)

Parameters:
    input_file: Name of the input CSV file (the folder is added in the main)
    output_file: Name of the output CSV file (the folder is added in the main)
    fv: Forecast Variable - index of the column to be predicted
    fh: Forecast Horizon - number of future values to predict
    ph: Past History - number of past values to use for prediction
"""

import argparse
import pandas as pd
import numpy as np


def transform_time_series(input_file, output_file, fv, fh, ph):
    """
    Transform time series data for regression tasks.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        fv (int): Index of the forecast variable column
        fh (int): Forecast horizon (number of future values to predict)
        ph (int): Past history (number of past values to use)
    """
    # Read the input CSV file
    print(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Get column names
    columns = df.columns.tolist()
    
    # Validate FV parameter
    if fv < 0 or fv >= len(columns):
        raise ValueError(f"FV index {fv} is out of range. Valid range: 0-{len(columns)-1}")
    
    # Get the name of the forecast variable
    fv_name = columns[fv]
    print(f"Forecast variable: {fv_name} (column index {fv})")
    
    # Create empty list to store transformed data
    transformed_data = []
    
    # For each possible starting point in the time series
    for i in range(len(df) - (ph + fh) + 1):
        # Get the window of past history for all variables
        past_window = df.iloc[i:i+ph].values
        
        # Get the window of future values for the forecast variable
        future_window = df.iloc[i+ph:i+ph+fh, fv].values
        
        # Create a row for the transformed data
        row = []
        
        # Add past values for all variables (flattened)
        for j in range(ph):
            row.extend(past_window[j])
        
        # Add future values for the forecast variable
        row.extend(future_window)
        
        # Add the row to the transformed data
        transformed_data.append(row)
    
    # Create column names for the transformed data
    transformed_columns = []
    
    # Add column names for past values of all variables
    for t in range(ph):
        for col in columns:
            transformed_columns.append(f"{col}_t-{ph-t}")
    
    # Add column names for future values of the forecast variable
    for t in range(fh):
        transformed_columns.append(f"{fv_name}_t+{t+1}")
    
    # Create a DataFrame with the transformed data
    transformed_df = pd.DataFrame(transformed_data, columns=transformed_columns)
    
    # Save the transformed data to the output CSV file
    print(f"Saving transformed data to: {output_file}")
    transformed_df.to_csv(output_file, index=False)
    print(f"Transformation complete. Created {len(transformed_df)} samples.")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Transform time series data for regression tasks.')
    parser.add_argument('input_file', type=str, help='Path to input CSV file')
    parser.add_argument('output_file', type=str, help='Path to output CSV file')
    parser.add_argument('--fv', type=int, required=True, help='Forecast Variable - index of the column to be predicted')
    parser.add_argument('--fh', type=int, required=True, help='Forecast Horizon - number of future values to predict')
    parser.add_argument('--ph', type=int, required=True, help='Past History - number of past values to use for prediction')
    
    args = parser.parse_args()
    
    # Transform the time series data
    transform_time_series("input_csv_files/"+args.input_file, "output_csv_files/"+args.output_file, args.fv, args.fh, args.ph)


if __name__ == '__main__':
    main()
