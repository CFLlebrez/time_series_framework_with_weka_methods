#!/usr/bin/env python3
"""
Script principal para integrar la transformación de series temporales
y la selección de atributos (opcional).

Nuevo flujo:
1. Primero se transforma el dataset (lags + horizontes).
2. Después, si se indica, se aplica selección de atributos sobre el CSV transformado.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature_selection import select_features
from feature_selection.visualization import generate_comprehensive_report


def main():
    # Parsear argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Framework para series temporales con selección de atributos.')
    
    # Argumentos para el archivo de entrada/salida
    parser.add_argument('input_file', type=str, help='Ruta al archivo CSV de entrada')
    parser.add_argument('output_dir', type=str, help='Directorio para guardar los resultados')
    
    # Argumentos para la transformación de series temporales
    parser.add_argument('--fv', type=int, required=True, help='Forecast Variable - índice de la columna a predecir')
    parser.add_argument('--fh', type=int, required=True, help='Forecast Horizon - número de valores futuros a predecir')
    parser.add_argument('--ph', type=int, required=True, help='Past History - número de valores pasados a utilizar')
    
    # Argumentos para columna temporal
    parser.add_argument('--time_col', type=str, required=True,
                        help='Nombre de la columna temporal (ej. fecha, date, timestamp)')
    
    # Argumentos para la selección de atributos
    parser.add_argument('--feature_selection', action='store_true', help='Aplicar selección de atributos')
    parser.add_argument('--fs_method', type=str, default='random_forest',
                        choices=['pearson', 'ccf', 'mutual_info', 'random_forest', 
                                 'lasso', 'elastic_net', 'rfe', 'granger', 'pca', 
                                 'spectral', 'sequential', 'genetic'],
                        help='Método de selección de atributos')
    parser.add_argument('--fs_n_features', type=int, default=None,
                        help='Número de atributos a seleccionar')
    parser.add_argument('--fs_threshold', type=float, default=None,
                        help='Umbral para la selección de atributos')
    
    args = parser.parse_args()

    input_folder_dir = os.path.join("input_csv_files", args.input_file)
    output_folder_dir = os.path.join("results", args.output_dir)
    os.makedirs(output_folder_dir, exist_ok=True)
    
    # Cargar datos originales (solo para identificar variable objetivo)
    print(f"Cargando datos desde {input_folder_dir}...")
    df = pd.read_csv(input_folder_dir)

    # Verificar que la columna temporal existe
    if args.time_col not in df.columns:
        raise ValueError(f"La columna temporal '{args.time_col}' no existe en el CSV. "
                         f"Columnas disponibles: {df.columns.tolist()}")
    
    # Forzar parseo a datetime
    df[args.time_col] = pd.to_datetime(df[args.time_col], errors='coerce')
    if df[args.time_col].isna().all():
        raise ValueError(f"No se pudo convertir la columna '{args.time_col}' a formato datetime.")
    
    columns = df.columns.tolist()
    
    if args.fv < 0 or args.fv >= len(columns):
        raise ValueError(f"FV index {args.fv} está fuera de rango. Rango válido: 0-{len(columns)-1}")
    
    target_col = columns[args.fv]
    print(f"Variable objetivo: {target_col} (índice {args.fv})")
    
    # --- Paso 1: Transformación de series temporales ---
    print("\nEjecutando transformación de series temporales...")
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from time_series_transformer_optimized import transform_time_series
    
    output_file = os.path.join(output_folder_dir, f'transformed_data_fv{args.fv}_fh{args.fh}_ph{args.ph}.csv')
    
    if not os.path.isfile(output_file):  # ✅ corregido: solo crea si no existe
        transform_time_series(
            input_folder_dir,
            output_file,
            args.fv,
            args.fh,
            args.ph,
            args.fv,         # original_fv = mismo índice en este punto
            args.time_col    # ✅ se pasa la columna temporal explícitamente
        )
        print(f"\nTransformación completada. Resultados guardados en: {output_file}")
    else:
        print(f"\nEl fichero {output_file} ya estaba creado.")
    
    transformed_csv = output_file
    
    # --- Paso 2: Selección de atributos (opcional) ---
    if args.feature_selection:
        print(f"\nAplicando selección de atributos usando método '{args.fs_method}'...")
        
        fs_results = select_features(
            transformed_csv,
            os.path.join(output_folder_dir, f'feature_selection_{args.fs_method}'),
            target_col,
            method=args.fs_method,
            n_features=args.fs_n_features,
            threshold=args.fs_threshold,
            max_lag=0,           # No se prueban lags, se hace selección sobre el csv transformado
            time_col=args.time_col  # ✅ ahora se informa la columna temporal
        )
        
        selected_features = fs_results['selected_features']
        print(f"\nCaracterísticas seleccionadas ({len(selected_features)}):")
        for i, feature in enumerate(selected_features, 1):
            importance = fs_results['feature_importances'].get(feature, 'N/A')
            if isinstance(importance, (int, float)):
                print(f"{i}. {feature} (importancia: {importance:.4f})")
            else:
                print(f"{i}. {feature} (importancia: {importance})")
        
        filtered_csv = fs_results['filtered_csv_path']
        print(f"\nCSV filtrado generado: {filtered_csv}")
    
    print("\nProceso completo finalizado.")


if __name__ == '__main__':
    main()
