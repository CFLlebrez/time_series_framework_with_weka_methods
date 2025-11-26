#!/usr/bin/env python3
"""
Time Series Transformer for Multivariate Prediction Tasks

This script transforms a CSV file containing multivariate time series data into a format
suitable for regression analysis. It takes a time series where each row is a temporal record
with multiple variables and transforms it into a format where each row contains:
- Past values of all variables (for context)
- Future values of the target variable (for prediction)

Parameters:
    input_file: Path to the input CSV file
    output_file: Path to the output CSV file
    fv: Forecast Variable - index of the column to be predicted
    fh: Forecast Horizon - number of future values to predict
    ph: Past History - number of past values to use for prediction
    time_col: Name of the column with timestamps/dates (will be used as index)
"""

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm


def transform_time_series(input_file, output_file, fv, fh, ph, original_fv, time_col=None):
    """
    Transform time series data for regression tasks with proper date alignment.
    The time column is treated separately and reinserted at the end.
    The FV (forecast variable) is kept in the past values, present (t0) and future horizon.
    """
    # Leer CSV
    print(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file)
    # Guardar indice por si se modifica
    fv_final = fv
    # Separar columna temporal si existe
    fechas = None
    if time_col is not None:
        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found in CSV. "
                             f"Available columns: {df.columns.tolist()}")
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        if df[time_col].isna().all():
            raise ValueError(f"Could not parse '{time_col}' as datetime.")
        fechas = df[time_col].copy()
        df = df.drop(columns=[time_col])
        fv_final-=1

    # Columnas de trabajo
    columns = df.columns.tolist()
    fv_name = columns[fv_final]
    if fv_final < 0 or fv_final >= len(columns):
        raise ValueError(f"FV index {fv} is out of range. Valid range: 0-{len(columns)-(fv_final==fv)}") #si fv_final está fuera de rango y es igual a fv, no se ha eliminado time_col y el rango es hasta len-1
    print(f"Forecast variable: {fv} (column index {fv})")

    transformed_data = []
    fechas_out = []

    print("Transforming data...")
    for i in range(len(df) - (ph + fh) + 1):
        # Pasado de todas las variables (incluye FV)
        past_window = df.iloc[i:i+ph].values
        # Futuro solo de la variable objetivo usando nombre
        future_window = df.iloc[i+ph:i+ph+fh][fv_name].values
        # Valor actual de la FV (t0)
        current_fv = df.iloc[i+ph-1][fv_name]

        row = []
        for j in range(ph):
            row.extend(past_window[j])  # Pasado de todas las variables
        row.append(current_fv)          # Valor actual de la FV
        row.extend(future_window)       # Horizontes futuros

        transformed_data.append(row)

        # Fecha asociada al instante t = última del pasado
        if fechas is not None:
            fechas_out.append(fechas.iloc[i+ph-1])

    # Construir nombres de columnas
    transformed_columns = []
    for t in range(ph):
        for col in columns:  # Incluye FV
            transformed_columns.append(f"{col}_t-{ph-t}")
    transformed_columns.append(f"{fv_name}")  # FV actual
    for t in range(fh):
        transformed_columns.append(f"{fv_name}_t+{t+1}")  # Horizontes futuros

    # Crear DataFrame
    transformed_df = pd.DataFrame(transformed_data, columns=transformed_columns)

    # Insertar columna temporal al principio
    if fechas is not None:
        transformed_df.insert(0, time_col, fechas_out)

    # Guardar CSV
    transformed_df.to_csv(output_file, index=False)

    print(f"Transformation complete. Created {len(transformed_df)} samples.")
    print(f"Saved to: {output_file}")
