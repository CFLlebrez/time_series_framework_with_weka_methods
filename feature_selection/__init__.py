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
import json
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
            method=kwargs.get('sklearn_method', 'selectkbest'),
            n_features=n_features,
            percentile=kwargs.get('percentile'),
            score_func_name=kwargs.get('score_func_name', 'f_regression'),
            strategy=kwargs.get('strategy', 'k_best'),
            param=kwargs.get('param'),
            threshold=kwargs.get('sklearn_threshold', 0.0),
            verbose=kwargs.get('verbose', False)
        )
    
    # Métodos inspirados en Weka
    elif method == 'weka_inspired':
        return create_weka_inspired_selector(
            method=kwargs.get('weka_inspired_method', 'cfs'),
            n_features=n_features,
            threshold=kwargs.get('weka_threshold', 0.0),
            #CFS
            max_backtrack=kwargs.get('max_backtrack', 5),
            #InfoGain
            n_bins=kwargs.get('n_bins', 10),
            discretize=kwargs.get('discretize', True),
            #ReliefF
            n_neighbors=kwargs.get('n_neighbors', 10),
            sample_size=kwargs.get('sample_size', None),
            discrete_threshold=kwargs.get('discrete_threshold', 10),
            n_jobs=kwargs.get('n_jobs', 1),
            pearson_prefilter=kwargs.get('pearson_prefilter', 0.0),
            verbose=kwargs.get('verbose', False)
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




def generate_feature_importance_report(importances, selected_list, output_dir, prefix):
    """Versión corregida para evitar warnings de paleta en Seaborn."""
    report_df = pd.DataFrame({
        'Feature': importances.index,
        'Importance': importances.values,
        'Selected': [f in selected_list for f in importances.index]
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    
    # Definimos la paleta de forma dinámica según lo que haya en el set de datos
    unique_selected = report_df.head(30)['Selected'].unique()
    palette = {True: 'darkblue', False: 'lightgray'}
    
    sns.barplot(x='Importance', y='Feature', data=report_df.head(30), 
                hue='Selected', palette=palette, legend=False)
    
    plt.title(f'Importancia de Características - {prefix}')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{prefix}_plot.png")
    plt.savefig(plot_path)
    plt.close()
    
    report_path = os.path.join(output_dir, f"{prefix}_report.csv")
    report_df.to_csv(report_path, index=False)
    return report_path, plot_path

def save_selection_json(output_path, method_name, selected_features, target_cols, elapsed_time, input_file):
    """
    Genera el archivo de metadatos para el evaluador.
    
    Args:
        output_path (str): Ruta para guardar el JSON.
        method_name (str): nombre del método
        selected_features (list): Lista de características seleccionadas.
        target_cols (list): Lista de variable objetivo y steps (si aplica)
        elapsed_time (time): tiempo de selección
        input_file (str): Ruta del fichero de entrada.
        
    Returns:
        DataFrame: Datos filtrados."""
    
    metadata = {
        "experiment_info": {
            "method": method_name,
            "input_file": os.path.basename(input_file),
            "date": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        "selected_features": list(selected_features),
        "target_columns": list(target_cols),
        "selection_time_seconds": round(elapsed_time, 4)
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)


def select_features(input_file, output_dir, target_col, method='random_forest', 
                   n_features=None, threshold=None, time_col=None, include_target=False,
                   generate_report=True, generate_filtered_csv=True, **kwargs):
    """
    Función principal para selección de características con control estricto de n_features.
    """
    import time
    verbose = kwargs.get('verbose', True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Carga de datos
    df = pd.read_csv(input_file)
    fechas = None
    if time_col and time_col in df.columns:
        fechas = df[time_col].copy()
        df = df.drop(columns=[time_col])

    # Preparar X e y
    X, y = prepare_data_for_selection(df, target_col, include_target=include_target)
    
    # Crear el selector
    selector = create_feature_selector(method, n_features, threshold, verbose=verbose, **kwargs)
    
    if verbose:
        print(f"Creando selector de características usando método '{method}'...")
        print("Ajustando selector de características...")
    
    # Medir tiempo de ejecución del ajuste original
    start_time = time.time()
    selector.fit(X, y)
    elapsed = time.time() - start_time
    
    # 1. Obtener características iniciales y sus importancias
    selected_features = list(selector.get_selected_features())
    importances = selector.get_feature_importances()
    print(f"DEBUG: Variables detectadas por el selector: {len(selected_features)}")
    print(f"DEBUG: ¿Existen importancias?: {len(importances) > 0}")
    print(f"DEBUG: Valor de n_features pedido: {n_features}")
    # 2. CONTROL ESTRICTO DEL NÚMERO DE ATRIBUTOS (Post-procesamiento)
    # Si el método devuelve más de lo pedido (común en Lasso o RF), recortamos manualmente
    if n_features is not None and len(selected_features) > n_features:
        if verbose:
            print(f"[POST-PROCESS] El método devolvió {len(selected_features)} variables. "
                  f"Recortando a las {n_features} más importantes según su peso.")
        
        # Crear un diccionario de importancia absoluta para las variables seleccionadas
        feat_imp = {f: abs(importances.get(f, 0)) for f in selected_features}
        
        # Ordenar de mayor a menor y tomar exactamente n_features
        selected_features = sorted(feat_imp, key=feat_imp.get, reverse=True)[:n_features]

    if verbose:
        print(f"Resultado final: {len(selected_features)} características seleccionadas.")

    # Definir etiqueta del método para archivos
    if method == 'sklearn_filter':
        method_label = f"sklearn_filter_{kwargs.get('sklearn_method')}"
    elif method == 'weka_inspired':
        method_label = f"weka_inspired_{kwargs.get('weka_inspired_method')}"
    else:
        method_label = method

    # 3. Generar informe visual y CSV de importancias
    report_path, plot_path = None, None
    if generate_report:
        report_path, plot_path = generate_feature_importance_report(
            importances, selected_features, output_dir, prefix=f"{method_label}_feature_importance"
        )
    
    # 4. Generar CSV filtrado (Opcional)
    filtered_csv_path = None
    if generate_filtered_csv:
        if verbose:
            print("Generando CSV filtrado con características seleccionadas + variable objetivo...")
        filtered_csv_path = os.path.join(output_dir, f"filtered_data_{method}.csv")
        
        # Subconjunto con features seleccionadas finales
        X_selected = X[selected_features]
        df_filtered = pd.concat([X_selected, y], axis=1)

        if fechas is not None:
            df_filtered.insert(0, time_col, fechas)

        # Si deseas guardar el CSV físicamente, descomenta la siguiente línea:
        # df_filtered.to_csv(filtered_csv_path, index=False)

    # 5. Identificar las columnas target (instantes futuros) para el JSON
    future_targets = [c for c in df.columns if c.startswith(f"{target_col}_t+")]
    all_targets = [target_col] + future_targets 

    # 6. Guardar el JSON de metadatos (crucial para el PredictiveEvaluator)
    # IMPORTANTE: Ya no llamamos a selector.fit() aquí para no perder el recorte
    json_path = os.path.join(output_dir, f"selection_metadata_{method}.json")
    save_selection_json(json_path, method_label, selected_features, all_targets, elapsed, input_file)
    
    return {
        'selected_features': selected_features,
        'target_columns': all_targets,
        'json_metadata_path': json_path,
        'n_selected': len(selected_features),
        'feature_importances': importances,
        'report_path': report_path,
        'plot_path': plot_path,
        'filtered_csv_path': filtered_csv_path
    }