#!/usr/bin/env python3
"""
Script principal para integrar la selección de atributos con la transformación de series temporales.

Este script combina la funcionalidad de selección de atributos con la transformación
de series temporales para crear un flujo de trabajo completo.
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
    parser.add_argument('--fs_max_lag', type=int, default=None,
                        help='Máximo lag a considerar (por defecto, igual a ph)')
    
    args = parser.parse_args()
    input_folder_dir = "input_csv_files/"+args.input_file
    output_folder_dir = "output_csv_files/"+args.output_dir
    # Crear directorio de salida si no existe
    os.makedirs(output_folder_dir, exist_ok=True)
    
    # Cargar datos
    print(f"Cargando datos desde {input_folder_dir}...")
    df = pd.read_csv(input_folder_dir)
    
    # Obtener nombre de la columna objetivo
    columns = df.columns.tolist()
    if args.fv < 0 or args.fv >= len(columns):
        raise ValueError(f"FV index {args.fv} está fuera de rango. Rango válido: 0-{len(columns)-1}")
    target_col = columns[args.fv]
    print(f"Variable objetivo: {target_col} (índice {args.fv})")
    
    # Determinar máximo lag para selección de atributos
    max_lag = args.fs_max_lag if args.fs_max_lag is not None else args.ph
    
    # Aplicar selección de atributos si se solicita
    if args.feature_selection:
        print(f"\nAplicando selección de atributos usando método '{args.fs_method}'...")
        
        # Ejecutar selección de atributos
        fs_results = select_features(
            input_folder_dir,
            os.path.join(output_folder_dir, 'feature_selection'),
            target_col,
            method=args.fs_method,
            n_features=args.fs_n_features,
            threshold=args.fs_threshold,
            max_lag=max_lag
        )
        
        # Mostrar características seleccionadas
        selected_features = fs_results['selected_features']
        print(f"\nCaracterísticas seleccionadas ({len(selected_features)}):")
        for i, feature in enumerate(selected_features, 1):
            importance = fs_results['feature_importances'].get(feature, 'N/A')
            if isinstance(importance, (int, float)):
                print(f"{i}. {feature} (importancia: {importance:.4f})")
            else:
                print(f"{i}. {feature} (importancia: {importance})")
        
        # Obtener ruta al CSV filtrado
        filtered_csv = fs_results['filtered_csv_path']
        print(f"\nCSV filtrado generado: {filtered_csv}")
        
        # Usar el CSV filtrado para la transformación
        input_for_transform = filtered_csv
    else:
        # Usar el CSV original para la transformación
        input_for_transform = input_folder_dir
    
    # Ejecutar transformación de series temporales
    print("\nEjecutando transformación de series temporales...")
    
    # Importar el script de transformación
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from time_series_transformer_optimized import transform_time_series
    
    # Definir archivo de salida para la transformación
    output_file = os.path.join(output_folder_dir, 'transformed_data.csv')
    
    # Ejecutar transformación
    transform_time_series(input_for_transform, output_file, args.fv, args.fh, args.ph)
    
    print(f"\nTransformación completada. Resultados guardados en: {output_file}")
    print("\nProceso completo finalizado.")


if __name__ == '__main__':
    main()
