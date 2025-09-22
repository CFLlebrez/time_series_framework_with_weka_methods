#!/usr/bin/env python3
"""
Módulo principal para la selección de atributos en series temporales.

Este módulo integra todos los métodos de selección de atributos y proporciona
una interfaz unificada para su uso.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Importar selectores de características
from .correlation_based import create_correlation_selector
from .model_based import create_model_selector
from .time_series_specific import create_time_series_selector
from .automatic_selection import create_automatic_selector
from .sklearn_filter import create_sklearn_filter_selector
from .weka_inspired import create_weka_inspired_selector


def create_feature_selector(method, n_features=None, threshold=None, **kwargs):
    """
    Crea un selector de características según el método especificado.
    
    Args:
        method (str): Método de selección.
        n_features (int, optional): Número de características a seleccionar.
        threshold (float, optional): Umbral para la selección de características.
        **kwargs: Argumentos adicionales específicos para cada método.
        
    Returns:
        BaseFeatureSelector: El selector de características correspondiente.
    """
    # Métodos basados en correlación
    if method in ['pearson', 'ccf', 'mutual_info']:
        return create_correlation_selector(method, n_features, threshold, **kwargs)
    
    # Métodos basados en modelos
    elif method in ['random_forest', 'lasso', 'elastic_net', 'rfe']:
        return create_model_selector(method, n_features, threshold, **kwargs)
    
    # Métodos específicos para series temporales
    elif method in ['granger', 'pca', 'spectral']:
        return create_time_series_selector(method, n_features, threshold, **kwargs)
    
    # Métodos de selección automática
    elif method in ['sequential', 'genetic']:
        return create_automatic_selector(method, n_features, **kwargs)
    
    # Métodos de filtro de scikit-learn
    elif method == 'sklearn_filter':
        return create_sklearn_filter_selector(
            method=kwargs.get('method', 'selectkbest'),
            n_features=n_features,
            percentile=kwargs.get('percentile'),
            score_func_name=kwargs.get('score_func_name', 'f_regression'),
            strategy=kwargs.get('strategy', 'k_best'),
            param=kwargs.get('param'),
            threshold=kwargs.get('threshold', 0.0),
            verbose=kwargs.get('verbose', False)
        )
    
    # Métodos inspirados en Weka
    elif method == 'weka_inspired':
        return create_weka_inspired_selector(
            method=kwargs.get('method', 'cfs'),
            n_features=n_features,
            threshold=threshold,
            verbose=kwargs.get('verbose', False),
            **kwargs
        )
    
    else:
        raise ValueError(f"Método desconocido: {method}")


def prepare_data_for_selection(df, target_col, include_target=False):
    """
    Prepara los datos para la selección de características, 
    eliminando la variable objetivo y sus horizontes futuros de X.

    Args:
        df (DataFrame): DataFrame ya transformado (con lags y horizontes).
        target_col (str): Nombre de la columna objetivo.
        time_col (str, optional): Columna de tiempo.

    Returns:
        tuple: (X, y) donde X son las variables predictoras y y es la variable objetivo.
    """
    data = df.copy()

    # Identificar columnas a excluir
    cols_to_exclude = []
    if not include_target:
        cols_to_exclude.append(target_col)

    # Excluir también horizontes futuros (target_col_t+N)
    future_cols = [c for c in data.columns if c.startswith(f"{target_col}_t+")]
    cols_to_exclude.extend(future_cols)
    # Construir X e y
    X = data.drop(columns=cols_to_exclude)
    y = data[target_col].copy()

    return X, y




def generate_feature_importance_report(selector, output_dir, prefix='feature_importance'):
    """
    Genera un informe de importancia de características y visualizaciones.
    
    Args:
        selector (BaseFeatureSelector): Selector de características ajustado.
        output_dir (str): Directorio para guardar los resultados.
        prefix (str): Prefijo para los nombres de archivo.
        
    Returns:
        tuple: (report_path, plot_path) rutas a los archivos generados.
    """
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Obtener importancias
    importances = selector.get_feature_importances()
    
    # Ordenar por importancia
    sorted_importances = importances.sort_values(ascending=False)
    
    # Crear DataFrame para el informe
    report_df = pd.DataFrame({
        'Feature': sorted_importances.index,
        'Importance': sorted_importances.values,
        'Selected': [feature in selector.get_selected_features() for feature in sorted_importances.index]
    })
    
    # Guardar informe
    report_path = os.path.join(output_dir, f"{prefix}_report.csv")
    report_df.to_csv(report_path, index=False)
    
    # Generar visualización
    plt.figure(figsize=(12, 8))
    
    # Crear gráfico de barras
    ax = sns.barplot(x='Importance', y='Feature', data=report_df.head(30), 
                    hue='Selected', palette=['lightgray', 'darkblue'])
    
    # Añadir títulos y etiquetas
    plt.title('Importancia de Características')
    plt.xlabel('Importancia')
    plt.ylabel('Característica')
    plt.tight_layout()
    
    # Guardar visualización
    plot_path = os.path.join(output_dir, f"{prefix}_plot.png")
    plt.savefig(plot_path)
    plt.close()
    
    return report_path, plot_path


def filter_data_with_selected_features(X, selected_features, output_file=None):
    """
    Filtra los datos para incluir solo las características seleccionadas.
    
    Args:
        X (DataFrame): Datos originales.
        selected_features (list): Lista de características seleccionadas.
        output_file (str, optional): Ruta para guardar el CSV filtrado.
        
    Returns:
        DataFrame: Datos filtrados.
    """
    # Filtrar datos
    X_filtered = X[selected_features]
    
    # Guardar CSV si se especifica ruta
    if output_file:
        X_filtered.to_csv(output_file, index=False)
    
    return X_filtered


def select_features(input_file, output_dir, target_col, method='random_forest', 
                   n_features=None, threshold=None, time_col=None, include_target=False,
                   generate_report=True, generate_filtered_csv=True, **kwargs):
    """
    Función principal para selección de características en series temporales.
    (Se asume que los lags ya están creados antes de esta fase).
    """
    verbose = kwargs.get('verbose', True)
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar datos transformados
    if verbose:
        print(f"Cargando datos desde {input_file}...")
    df = pd.read_csv(input_file)

    # Separar columna temporal si existe
    fechas = None
    if time_col and time_col in df.columns:
        fechas = df[time_col].copy()
        df = df.drop(columns=[time_col])
    
    # Preparar datos (sin generar lags, ya vienen creados)
    if verbose:
        print("Preparando datos para selección (sin recalcular lags)...")

    X, y = prepare_data_for_selection(df, target_col, include_target=include_target)
    
    # Crear selector
    if verbose:
        print(f"Creando selector de características usando método '{method}'...")
    selector = create_feature_selector(method, n_features, threshold, verbose=verbose, **kwargs)
    
    # Ajustar selector
    if verbose:
        print("Ajustando selector de características...")
    selector.fit(X, y)
    
    # Obtener características seleccionadas
    selected_features = selector.get_selected_features()
    if verbose:
        print(f"Seleccionadas {len(selected_features)} características de {X.shape[1]}")
    
    # Informe
    report_path, plot_path = None, None
    if generate_report:
        if verbose:
            print("Generando informe de importancia de características...")
        report_path, plot_path = generate_feature_importance_report(
            selector, output_dir, prefix=f"{method}_feature_importance"
        )
    
    # CSV filtrado
    filtered_csv_path = None
    if generate_filtered_csv:
        if verbose:
            print("Generando CSV filtrado con características seleccionadas + variable objetivo...")
        filtered_csv_path = os.path.join(output_dir, f"filtered_data_{method}.csv")
        
        # Subconjunto con features seleccionadas
        X_selected = X[selected_features]
        df_filtered = pd.concat([X_selected, y], axis=1)

        # Reinsertar columna temporal si estaba presente
        if fechas is not None:
            df_filtered.insert(0, time_col, fechas)

        df_filtered.to_csv(filtered_csv_path, index=False)
    
    return {
        'selected_features': selected_features,
        'n_selected': len(selected_features),
        'feature_importances': selector.get_feature_importances(),
        'report_path': report_path,
        'plot_path': plot_path,
        'filtered_csv_path': filtered_csv_path
    }



