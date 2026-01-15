#!/usr/bin/env python3
"""
Script para probar los métodos de selección de atributos inspirados en Weka.

Este script demuestra el uso de los diferentes métodos inspirados en Weka
implementados en el módulo feature_selection.weka_inspired.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature_selection import select_features, prepare_lagged_data
from feature_selection.visualization import generate_comprehensive_report

# Configuración
INPUT_FILE = 'sample_data.csv'
OUTPUT_DIR = './weka_inspired_results'
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

# Probar diferentes métodos inspirados en Weka
methods = [
    # CFS (Correlation-based Feature Selection)
    {'name': 'weka_inspired', 'params': {
        'method': 'cfs', 
        'n_features': 10,
        'max_backtrack': 5
    }},
    
    # InfoGain (Information Gain)
    {'name': 'weka_inspired', 'params': {
        'method': 'infogain', 
        'n_features': 10,
        'discretize': True,
        'n_bins': 10
    }},
    
    # ReliefF
    {'name': 'weka_inspired', 'params': {
        'method': 'relieff', 
        'n_features': 10,
        'n_neighbors': 10,
        'sample_size': 100  # Usar muestra para acelerar el proceso
    }}
]

# Ejecutar cada método y generar resultados
results = {}

for method_info in methods:
    method_name = method_info['name']
    params = method_info['params']
    
    # Crear un nombre descriptivo para el directorio de resultados
    if 'method' in params:
        result_dir = f"{method_name}_{params['method']}"
    else:
        result_dir = method_name
    
    print(f"\n{'='*50}")
    print(f"Ejecutando método: {method_name}")
    print(f"Parámetros: {params}")
    
    try:
        # Ejecutar selección de atributos
        method_results = select_features(
            INPUT_FILE,
            os.path.join(OUTPUT_DIR, result_dir),
            TARGET_COL,
            method=method_name,
            max_lag=MAX_LAG,
            **params
        )
        
        # Guardar resultados
        results[result_dir] = method_results
        
        # Mostrar características seleccionadas
        selected_features = method_results['selected_features']
        print(f"\nCaracterísticas seleccionadas ({len(selected_features)}):")
        for i, feature in enumerate(selected_features, 1):
            importance = method_results['feature_importances'].get(feature, 'N/A')
            if isinstance(importance, (int, float)):
                print(f"{i}. {feature} (importancia: {importance:.4f})")
            else:
                print(f"{i}. {feature} (importancia: {importance})")
        
        print(f"\nResultados guardados en: {os.path.join(OUTPUT_DIR, result_dir)}")
        
    except Exception as e:
        print(f"Error al ejecutar método {method_name} con parámetros {params}: {e}")

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
print("\nPrueba de métodos inspirados en Weka completada.")
