#!/usr/bin/env python3
"""
Script principal para integrar la transformación de series temporales
y la selección de atributos (opcional).

Nuevo flujo:
1. Primero se transforma el dataset (lags + horizontes).
2. Después, si se indica, se aplica selección de atributos sobre el CSV transformado.
"""

import json
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature_selection import select_features
from evaluation import PredictiveEvaluator


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
                                 'spectral', 'sequential', 'genetic', 'sklearn_filter', 'weka_inspired'],
                        help='Método de selección de atributos')
    parser.add_argument('--sklearn_method', type=str, default='selectkbest', 
                        choices=['selectkbest', 'selectpercentile', 
                                 'genericunivariateselect', 'variancethreshold'],
                        help='Método de selección de filtro de sklearn')
    parser.add_argument('--weka_inspired_method', type=str, default='cfs', 
                        choices=['cfs', 'relieff', 'infogain'],
                        help='Método de selección de filtro inspirado en weka')
    parser.add_argument('--fs_n_features', type=int, default=None,
                        help='Número de atributos a seleccionar')
    parser.add_argument('--fs_threshold', type=float, default=None,
                        help='Umbral para la selección de atributos')
    # Argumentos weka
    parser.add_argument('--infogain_discretize', action='store_true', 
                        help='Discretizar la variable objetivo InfoGain, depende de cuantiles n_bins')
    parser.add_argument('--infogain_nbins', type=int, default=10,
                        help='Parámetro para InfoGain de Weka')
    parser.add_argument('--fs_max_backtrack', type=int, default=5,
                        help='Parámetro para CFS de Weka')
    parser.add_argument('--relieff_sample_size', type=int, default=None,
                        help='Parámetro para CFS de Weka')
    parser.add_argument('--relieff_n_neighbors', type=int, default=10,
                        help='Parámetro para ReliefF de Weka')
    parser.add_argument('--relieff_discrete_threshold', type=int, default=10,
                        help='Parámetro para ReliefF de Weka')
    parser.add_argument('--relieff_n_jobs', type=int, default=1,
                        help='Parámetro para ReliefF de Weka')
    # Argumentos lasso and elastic net
    parser.add_argument('--alpha', type=float, default=None,
                        help='Parámetro para métodos Lasso y ElasticNet')
    # Argumentos sklearn
    parser.add_argument('--percentile', type=float, default=None,
                        help='Parámetro para SKlearn select percentile')
    parser.add_argument('--strategy', type=str, default='fpr',
                        choices=['fpr', 'fdr', 'fwe'],
                        help='Parámetro para SKlearn estrategia para generic univariate select')
    parser.add_argument('--param', type=float, default=1e-50,
                        help='Parámetro para SKlearn generic univariate select')
    # Argumento para evaluacion
    parser.add_argument('--evaluation', action='store_true', help='Aplicar evaluación del método')
    
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
        raise ValueError(f"FV index {args.fv} está fuera de rango. Rango válido: 0-{len(columns)-1}, columnas {columns}")
    
    target_col = columns[args.fv]
    print(f"Variable objetivo: {target_col} (índice {args.fv})")
    
    # --- Paso 1: Transformación de series temporales ---
    print("\nEjecutando transformación de series temporales...")
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from time_series_transformer_optimized import transform_time_series
    
    output_file = os.path.join(output_folder_dir, f'transformed_data_fv{args.fv}_fh{args.fh}_ph{args.ph}.csv')
    
    if not os.path.isfile(output_file):  #  corregido: solo crea si no existe
        transform_time_series(
            input_folder_dir,
            output_file,
            args.fv,
            args.fh,
            args.ph,
            args.fv,         # original_fv = mismo índice en este punto
            args.time_col    #  se pasa la columna temporal explícitamente
        )
        print(f"\nTransformación completada. Resultados guardados en: {output_file}")
    else:
        print(f"\nEl fichero {output_file} ya estaba creado.")
    
    transformed_csv = output_file
    
    # --- Paso 2: Selección de atributos (opcional) ---
    if args.feature_selection:
        print("\n" + "-"*40)
        print(f"\nAplicando selección de atributos usando método '{args.fs_method}'...")
        if args.fs_method in ['lasso', 'elastic_net']:
            fs_results = select_features(
                transformed_csv,
                os.path.join(output_folder_dir, f'feature_selection_{args.fs_method}'),
                target_col,
                method=args.fs_method,
                n_features=args.fs_n_features,
                threshold=args.fs_threshold,
                max_lag=0,           # No se prueban lags, se hace selección sobre el csv transformado
                time_col=args.time_col,  #  ahora se informa la columna temporal
                alpha=args.alpha if args.alpha else 1.0
            )
        elif args.fs_method in ['pearson', 'ccf', 'mutual_info']: # métodos basados en correlación
            fs_results = select_features(
                transformed_csv,
                os.path.join(output_folder_dir, f'feature_selection_{args.fs_method}'),
                target_col,
                method=args.fs_method,
                n_features=args.fs_n_features,
                threshold=args.fs_threshold,
                max_lag=0,           # No se prueban lags, se hace selección sobre el csv transformado
                time_col=args.time_col,  #  ahora se informa la columna temporal
            )
        elif args.fs_method=='sklearn_filter':
            fs_results = select_features(
                transformed_csv,
                os.path.join(output_folder_dir, f'feature_selection_{args.fs_method}_{args.sklearn_method}'),
                target_col,
                method=args.fs_method,
                sklearn_method=args.sklearn_method,
                n_features=args.fs_n_features,
                sklearn_threshold=args.fs_threshold,
                percentile=args.percentile,
                strategy=args.strategy,
                param=args.param,
                max_lag=0,           # No se prueban lags, se hace selección sobre el csv transformado
                time_col=args.time_col,  #  ahora se informa la columna temporal
            )
        elif args.fs_method=='weka_inspired':
            fs_results = select_features(
                transformed_csv,
                os.path.join(output_folder_dir, f'feature_selection_{args.fs_method}_{args.weka_inspired_method}'),
                target_col,
                method=args.fs_method,
                weka_inspired_method=args.weka_inspired_method,
                n_features=args.fs_n_features,
                weka_threshold=args.fs_threshold,
                max_backtrack=args.fs_max_backtrack,
                discretize=args.infogain_discretize,
                n_bins=args.infogain_nbins,
                n_neighbors=args.relieff_n_neighbors,
                sample_size=args.relieff_sample_size,
                discrete_threshold=args.relieff_discrete_threshold,
                n_jobs=args.relieff_n_jobs,
                max_lag=0,           # No se prueban lags, se hace selección sobre el csv transformado
                time_col=args.time_col,  #  ahora se informa la columna temporal
            )
        else: # Métodos sin parámetros específicos
            fs_results = select_features(
                transformed_csv,
                os.path.join(output_folder_dir, f'feature_selection_{args.fs_method}'),
                target_col,
                method=args.fs_method,
                n_features=args.fs_n_features,
                threshold=args.fs_threshold,
                max_lag=0,           # No se prueban lags, se hace selección sobre el csv transformado
                time_col=args.time_col,  #  ahora se informa la columna temporal
            )


        selected_features = fs_results['selected_features']
        target_columns = fs_results['target_columns']
        
        print(f"\nCaracterísticas seleccionadas ({len(selected_features)}):")
        for i, feature in enumerate(selected_features, 1):
            importance = fs_results['feature_importances'].get(feature, 'N/A')
            if isinstance(importance, (int, float)):
                print(f"{i}. {feature} (importancia: {importance:.4f})")
            else:
                print(f"{i}. {feature} (importancia: {importance})")
        
        json_path = fs_results['json_metadata_path']
        print(f"\n[INFO] Metadatos de selección guardados en: {json_path}")
        print(f"Prediciendo horizonte: {target_columns}")
    
    if args.evaluation:
        print("\n" + "-"*40)
        print("INICIANDO TUNING Y EVALUACIÓN")
        print("-"*40)
        
        try:
            from tuner import tune_knn_k
            
            # 1. Identificar el nombre específico del método para el reporte
            if args.fs_method == 'sklearn_filter':
                method_label = f"sklearn_{args.sklearn_method}"
            elif args.fs_method == 'weka_inspired':
                method_label = f"weka_{args.weka_inspired_method}"
            else:
                method_label = args.fs_method

            # 2. Cargar datos para Tuning (K del KNN)
            df_eval = pd.read_csv(transformed_csv)
            df_num = df_eval.select_dtypes(include=[np.number])
            
            with open(json_path, 'r') as f:
                meta = json.load(f)
            
            # Buscamos el mejor K usando todas las variables (Baseline) para ser justos
            target_cols = meta['target_columns']
            all_feats = [c for c in df_num.columns if c not in target_cols]
            
            best_k = tune_knn_k(df_num, all_feats, target_cols)
            
            # 3. Evaluación oficial con el K óptimo encontrado
            evaluator = PredictiveEvaluator(n_splits=5, n_neighbors=best_k)
            report = evaluator.run_full_evaluation(transformed_csv, json_path, args.time_col)
            
            # 4. Guardar y actualizar máster
            params_str = f"_fv{args.fv}_fh{args.fh}_ph{args.ph}"
            final_report_path = os.path.join(output_folder_dir, f"evaluation_report_{method_label}.csv")
            report.to_csv(final_report_path, index=False)
            
            master_path = evaluator.update_master_report(report, output_folder_dir, params_str)
            
            print(f"\n[OK] K óptimo utilizado: {best_k}")
            print(f"[OK] Master Report actualizado: {master_path}")
            
        except Exception as e:
            print(f"\n[ERROR] Error en evaluación: {e}")
            import traceback
            traceback.print_exc()
            
    print("\nProceso completo finalizado.")


if __name__ == '__main__':
    main()
