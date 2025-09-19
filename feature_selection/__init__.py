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


def prepare_lagged_data(df, target_col, max_lag, include_target=True):
    """
    Prepara datos con lags para selección de características.
    
    Args:
        df (DataFrame): DataFrame original.
        target_col (str): Nombre de la columna objetivo.
        max_lag (int): Máximo lag a considerar.
        include_target (bool): Si es True, incluye la variable objetivo en los predictores.
        
    Returns:
        tuple: (X_lagged, y) donde X_lagged contiene todas las variables con lags y y es la variable objetivo.
    """
    # Crear copia del DataFrame
    data = df.copy()

    # Obtener lista de columnas
    columns = data.columns.tolist()
    
    # Crear DataFrame para almacenar datos con lags
    X_lagged = pd.DataFrame(index=data.index)
    
    # Añadir lags para cada columna
    for col in columns:
        # Saltar la columna objetivo si no se debe incluir
        if col == target_col and not include_target:
            continue
        
        # Añadir lags
        for lag in range(1, max_lag + 1):
            X_lagged[f"{col}_lag{lag}"] = data[col].shift(lag)
    
    # Eliminar filas con NaN
    X_lagged = X_lagged.dropna()
    
    # Preparar variable objetivo (alineada con X_lagged)
    y = data.loc[X_lagged.index, target_col]
    
    return X_lagged, y


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
                   n_features=None, threshold=None, max_lag=10, include_target=True,
                   generate_report=True, generate_filtered_csv=True, **kwargs):
    """
    Función principal para selección de características en series temporales.
    
    Args:
        input_file (str): Ruta al archivo CSV de entrada.
        output_dir (str): Directorio para guardar los resultados.
        target_col (str): Nombre de la columna objetivo.
        method (str): Método de selección de características.
        n_features (int, optional): Número de características a seleccionar.
        threshold (float, optional): Umbral para la selección de características.
        max_lag (int): Máximo lag a considerar.
        include_target (bool): Si es True, incluye la variable objetivo en los predictores.
        generate_report (bool): Si es True, genera informe y visualizaciones.
        generate_filtered_csv (bool): Si es True, genera CSV filtrado.
        **kwargs: Argumentos adicionales específicos para cada método.
        
    Returns:
        dict: Diccionario con resultados y rutas a archivos generados.
    """
    verbose = kwargs.get('verbose', True)
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar datos
    if verbose:
        print(f"Cargando datos desde {input_file}...")
    df = pd.read_csv(input_file)
    
    # Preparar datos con lags
    if verbose:
        print(f"Preparando datos con lags (max_lag={max_lag})...")
    X_lagged, y = prepare_lagged_data(df, target_col, max_lag, include_target)
    
    # Crear selector de características
    if verbose:
        print(f"Creando selector de características usando método '{method}'...")
    selector = create_feature_selector(method, n_features, threshold, verbose=verbose, **kwargs)
    
    # Ajustar selector
    if verbose:
        print("Ajustando selector de características...")
    selector.fit(X_lagged, y)
    
    # Obtener características seleccionadas
    selected_features = selector.get_selected_features()
    
    if verbose:
        print(f"Seleccionadas {len(selected_features)} características de {X_lagged.shape[1]}")
    
    # Generar informe y visualizaciones
    report_path = None
    plot_path = None
    if generate_report:
        if verbose:
            print("Generando informe de importancia de características...")
        report_path, plot_path = generate_feature_importance_report(
            selector, output_dir, prefix=f"{method}_feature_importance"
        )
    
    # Generar CSV filtrado
        # Generar CSV filtrado
    filtered_csv_path = None
    if generate_filtered_csv:
        if verbose:
            print("Generando CSV filtrado con características seleccionadas + variable objetivo...")
        filtered_csv_path = os.path.join(output_dir, f"filtered_data_{method}.csv")
        
        # Subconjunto con features seleccionadas
        X_selected = X_lagged[selected_features]
        
        # Concatenar variable objetivo como última columna
        df_filtered = pd.concat([X_selected, y], axis=1)
        
                # 🔍 Comprobar si la primera columna del dataset original es fecha
        first_col = df.columns[0]
        try:
            parsed = pd.to_datetime(df[first_col], errors='coerce')
            is_datetime = parsed.notna().sum() > 0.9 * len(parsed)  # al menos 90% convertibles
        except Exception:
            is_datetime = False
        
        # Si es fecha, añadirla como primera columna en el CSV final
        if is_datetime:
            # Usar el mismo índice que X_lagged / y (para evitar desajuste de longitudes)
            date_aligned = df.loc[X_lagged.index, first_col]
            df_filtered.insert(0, first_col, date_aligned.values)

            if verbose:
                print(f"Columna de fecha detectada y preservada: '{first_col}'")
        
        # Guardar en CSV
        df_filtered.to_csv(filtered_csv_path, index=False)
    
    # Devolver resultados
    results = {
        'selected_features': selected_features,
        'n_selected': len(selected_features),
        'feature_importances': selector.get_feature_importances(),
        'report_path': report_path,
        'plot_path': plot_path,
        'filtered_csv_path': filtered_csv_path
    }
    
    return results

