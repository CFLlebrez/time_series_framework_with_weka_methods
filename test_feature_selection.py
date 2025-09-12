#!/usr/bin/env python3
"""
Script para probar los métodos de selección de atributos con datos de ejemplo.

Este script demuestra el uso de los diferentes métodos de selección de atributos
implementados en el módulo feature_selection.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature_selection import select_features, prepare_lagged_data
from feature_selection.visualization import generate_comprehensive_report

# Configuración
INPUT_FILE = '../sample_data.csv'
OUTPUT_DIR = './feature_selection_results'
TARGET_COL = 'temperatura'  # Columna objetivo para la predicción
MAX_LAG = 5                # Máximo lag a considerar

# Crear directorio de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cargar datos
print(f"Cargando datos desde {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

# Mostrar información sobre los datos
print("\nInformación del dataset:")
print(f"- Número de filas: {len(df)}")
print(f"- Número de columnas: {len(df.columns)}")
print(f"- Columnas: {', '.join(df.columns)}")
print(f"- Variable objetivo: {TARGET_COL}")

# Preparar datos con lags para visualización
print(f"\nPreparando datos con lags (max_lag={MAX_LAG})...")
X_lagged, y = prepare_lagged_data(df, TARGET_COL, MAX_LAG)
print(f"- Dimensiones de X_lagged: {X_lagged.shape}")
print(f"- Número de características: {X_lagged.shape[1]}")

# Probar diferentes métodos de selección de atributos
methods = [
    # Métodos basados en correlación
    {'name': 'pearson', 'params': {'n_features': 10}},
    {'name': 'ccf', 'params': {'n_features': 10, 'max_lag': MAX_LAG}},
    {'name': 'mutual_info', 'params': {'n_features': 10}},
    
    # Métodos basados en modelos
    {'name': 'random_forest', 'params': {'n_features': 10, 'n_estimators': 100}},
    {'name': 'lasso', 'params': {'n_features': 10, 'alpha': 0.01}},
    {'name': 'elastic_net', 'params': {'n_features': 10, 'alpha': 0.01, 'l1_ratio': 0.5}},
    
    # Métodos específicos para series temporales
    {'name': 'granger', 'params': {'n_features': 10, 'max_lag': MAX_LAG}},
    {'name': 'pca', 'params': {'n_features': 10}},
    {'name': 'spectral', 'params': {'n_features': 10, 'method': 'periodogram'}},
    
    # Métodos de selección automática
    {'name': 'sequential', 'params': {'n_features': 10, 'direction': 'forward'}},
    # Nota: El método genético es computacionalmente intensivo, usar con precaución
    # {'name': 'genetic', 'params': {'n_features': 10, 'population_size': 20, 'generations': 10}}
]

# Ejecutar cada método y generar resultados
results = {}

for method_info in methods:
    method_name = method_info['name']
    params = method_info['params']
    
    print(f"\n{'='*50}")
    print(f"Ejecutando método: {method_name}")
    print(f"Parámetros: {params}")
    
    try:
        # Ejecutar selección de atributos
        method_results = select_features(
            INPUT_FILE,
            os.path.join(OUTPUT_DIR, method_name),
            TARGET_COL,
            method=method_name,
            max_lag=MAX_LAG,
            **params
        )
        
        # Guardar resultados
        results[method_name] = method_results
        
        # Mostrar características seleccionadas
        selected_features = method_results['selected_features']
        print(f"\nCaracterísticas seleccionadas ({len(selected_features)}):")
        for i, feature in enumerate(selected_features, 1):
            importance = method_results['feature_importances'].get(feature, 'N/A')
            print(f"{i}. {feature} (importancia: {importance:.4f})")
        
        print(f"\nResultados guardados en: {os.path.join(OUTPUT_DIR, method_name)}")
        
    except Exception as e:
        print(f"Error al ejecutar método {method_name}: {e}")

# Generar informe comparativo
print("\n\nGenerando informe comparativo de métodos...")

# Crear DataFrame con resultados comparativos
comparison_data = []

for method_name, method_results in results.items():
    comparison_data.append({
        'Método': method_name,
        'Características seleccionadas': len(method_results['selected_features']),
        'Ruta del informe': method_results.get('report_path', 'N/A'),
        'Ruta del CSV filtrado': method_results.get('filtered_csv_path', 'N/A')
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_path = os.path.join(OUTPUT_DIR, 'method_comparison.csv')
comparison_df.to_csv(comparison_path, index=False)

print(f"Informe comparativo guardado en: {comparison_path}")
print("\nPrueba de métodos de selección de atributos completada.")
