#!/usr/bin/env python3
"""
Módulo de visualización para métodos de selección de atributos en series temporales.

Este módulo proporciona funciones para visualizar los resultados de la selección de atributos,
incluyendo gráficos de importancia, matrices de correlación, y más.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def plot_feature_importance(importances, selected_features=None, top_n=30, 
                           figsize=(12, 8), save_path=None, title='Importancia de Características'):
    """
    Visualiza la importancia de las características.
    
    Args:
        importances (Series): Serie con importancias de características.
        selected_features (list, optional): Lista de características seleccionadas.
        top_n (int, optional): Número de características principales a mostrar.
        figsize (tuple, optional): Tamaño de la figura.
        save_path (str, optional): Ruta para guardar la figura.
        title (str, optional): Título del gráfico.
        
    Returns:
        matplotlib.figure.Figure: La figura generada.
    """
    # Ordenar importancias
    sorted_importances = importances.sort_values(ascending=False)
    
    # Limitar a top_n
    if top_n is not None and len(sorted_importances) > top_n:
        sorted_importances = sorted_importances.iloc[:top_n]
    
    # Crear DataFrame para visualización
    df_plot = pd.DataFrame({
        'Feature': sorted_importances.index,
        'Importance': sorted_importances.values
    })
    
    # Añadir columna de selección si se proporcionan características seleccionadas
    if selected_features is not None:
        df_plot['Selected'] = df_plot['Feature'].isin(selected_features)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Crear gráfico de barras
    if selected_features is not None:
        ax = sns.barplot(x='Importance', y='Feature', data=df_plot, 
                        hue='Selected', palette=['lightgray', 'darkblue'])
        ax.legend(title='Seleccionada')
    else:
        ax = sns.barplot(x='Importance', y='Feature', data=df_plot)
    
    # Añadir títulos y etiquetas
    ax.set_title(title)
    ax.set_xlabel('Importancia')
    ax.set_ylabel('Característica')
    
    # Ajustar diseño
    plt.tight_layout()
    
    # Guardar figura si se especifica ruta
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_correlation_matrix(X, selected_features=None, figsize=(12, 10), 
                           save_path=None, title='Matriz de Correlación'):
    """
    Visualiza la matriz de correlación entre características.
    
    Args:
        X (DataFrame): Datos de entrada.
        selected_features (list, optional): Lista de características seleccionadas a destacar.
        figsize (tuple, optional): Tamaño de la figura.
        save_path (str, optional): Ruta para guardar la figura.
        title (str, optional): Título del gráfico.
        
    Returns:
        matplotlib.figure.Figure: La figura generada.
    """
    # Calcular matriz de correlación
    corr_matrix = X.corr()
    
    # Crear figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Crear mapa de calor
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
               square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    
    # Destacar características seleccionadas si se proporcionan
    if selected_features is not None:
        # Obtener índices de características seleccionadas
        selected_indices = [i for i, feature in enumerate(corr_matrix.columns) 
                           if feature in selected_features]
        
        # Destacar características seleccionadas
        for idx in selected_indices:
            ax.add_patch(plt.Rectangle((idx, 0), 1, corr_matrix.shape[0], 
                                      fill=False, edgecolor='red', lw=2))
            ax.add_patch(plt.Rectangle((0, idx), corr_matrix.shape[1], 1, 
                                      fill=False, edgecolor='red', lw=2))
    
    # Añadir título
    ax.set_title(title)
    
    # Ajustar diseño
    plt.tight_layout()
    
    # Guardar figura si se especifica ruta
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_feature_distribution(X, selected_features=None, max_features=10, 
                             figsize=(15, 10), save_path=None):
    """
    Visualiza la distribución de las características seleccionadas.
    
    Args:
        X (DataFrame): Datos de entrada.
        selected_features (list, optional): Lista de características a visualizar.
        max_features (int, optional): Número máximo de características a mostrar.
        figsize (tuple, optional): Tamaño de la figura.
        save_path (str, optional): Ruta para guardar la figura.
        
    Returns:
        matplotlib.figure.Figure: La figura generada.
    """
    # Seleccionar características a visualizar
    if selected_features is None:
        features_to_plot = X.columns[:max_features].tolist()
    else:
        features_to_plot = selected_features[:max_features]
    
    # Determinar número de filas y columnas para subplots
    n_features = len(features_to_plot)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Crear figura
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Aplanar array de axes si es necesario
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    
    # Graficar distribución para cada característica
    for i, feature in enumerate(features_to_plot):
        if i < len(axes):
            row = i // n_cols
            col = i % n_cols
            
            if n_rows == 1 and n_cols == 1:
                ax = axes[0]
            elif n_rows == 1 or n_cols == 1:
                ax = axes[i]
            else:
                ax = axes[row, col]
            
            # Graficar histograma y KDE
            sns.histplot(X[feature].dropna(), kde=True, ax=ax)
            
            # Añadir título
            ax.set_title(feature)
            
            # Ajustar etiquetas
            ax.set_xlabel('')
            if col == 0:
                ax.set_ylabel('Frecuencia')
            else:
                ax.set_ylabel('')
    
    # Ocultar axes no utilizados
    for i in range(n_features, len(axes.flatten())):
        fig.delaxes(axes.flatten()[i])
    
    # Añadir título general
    fig.suptitle('Distribución de Características Seleccionadas', fontsize=16)
    
    # Ajustar diseño
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Guardar figura si se especifica ruta
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_lag_importance(importances, max_lag=10, figsize=(12, 8), save_path=None):
    """
    Visualiza la importancia de diferentes lags para cada variable.
    
    Args:
        importances (Series): Serie con importancias de características.
        max_lag (int, optional): Máximo lag a considerar.
        figsize (tuple, optional): Tamaño de la figura.
        save_path (str, optional): Ruta para guardar la figura.
        
    Returns:
        matplotlib.figure.Figure: La figura generada.
    """
    # Extraer variables base y sus lags
    lag_pattern = '_lag'
    base_vars = set()
    for feature in importances.index:
        if lag_pattern in feature:
            base_var = feature.split(lag_pattern)[0]
            base_vars.add(base_var)
    
    # Crear DataFrame para almacenar importancias por variable y lag
    lag_importances = {}
    
    for base_var in base_vars:
        var_importances = []
        
        for lag in range(1, max_lag + 1):
            lag_feature = f"{base_var}{lag_pattern}{lag}"
            
            if lag_feature in importances:
                var_importances.append((lag, importances[lag_feature]))
            else:
                var_importances.append((lag, 0))
        
        if var_importances:
            lag_importances[base_var] = var_importances
    
    # Crear figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Graficar importancia por lag para cada variable
    for base_var, values in lag_importances.items():
        lags, imps = zip(*values)
        ax.plot(lags, imps, 'o-', label=base_var)
    
    # Añadir títulos y etiquetas
    ax.set_title('Importancia por Lag para cada Variable')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Importancia')
    ax.set_xticks(range(1, max_lag + 1))
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Ajustar diseño
    plt.tight_layout()
    
    # Guardar figura si se especifica ruta
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_parallel_coordinates(X, selected_features=None, n_samples=100, figsize=(15, 8), save_path=None):
    """
    Visualiza coordenadas paralelas para las características seleccionadas.
    
    Args:
        X (DataFrame): Datos de entrada.
        selected_features (list, optional): Lista de características a visualizar.
        n_samples (int, optional): Número de muestras a visualizar.
        figsize (tuple, optional): Tamaño de la figura.
        save_path (str, optional): Ruta para guardar la figura.
        
    Returns:
        matplotlib.figure.Figure: La figura generada.
    """
    # Seleccionar características a visualizar
    if selected_features is None:
        features_to_plot = X.columns.tolist()
    else:
        features_to_plot = selected_features
    
    # Seleccionar subconjunto de datos si hay muchas muestras
    if len(X) > n_samples:
        X_sample = X.sample(n_samples, random_state=42)
    else:
        X_sample = X
    
    # Seleccionar solo las características a visualizar
    X_plot = X_sample[features_to_plot].copy()
    
    # Normalizar datos para mejor visualización
    scaler = MinMaxScaler()
    X_normalized = pd.DataFrame(
        scaler.fit_transform(X_plot),
        columns=X_plot.columns,
        index=X_plot.index
    )
    
    # Crear figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Graficar coordenadas paralelas
    pd.plotting.parallel_coordinates(
        X_normalized.reset_index(), 'index', 
        colormap=plt.cm.tab20, ax=ax
    )
    
    # Añadir títulos y etiquetas
    ax.set_title('Coordenadas Paralelas de Características Seleccionadas')
    ax.set_xlabel('')
    ax.set_ylabel('Valor Normalizado')
    
    # Ocultar leyenda si hay muchas muestras
    if len(X_normalized) > 20:
        ax.get_legend().remove()
    
    # Ajustar diseño
    plt.tight_layout()
    
    # Guardar figura si se especifica ruta
    if save_path:
        plt.savefig(save_path)
    
    return fig


def generate_comprehensive_report(X, importances, selected_features, output_dir, prefix='feature_selection'):
    """
    Genera un informe completo con múltiples visualizaciones.
    
    Args:
        X (DataFrame): Datos de entrada.
        importances (Series): Serie con importancias de características.
        selected_features (list): Lista de características seleccionadas.
        output_dir (str): Directorio para guardar los resultados.
        prefix (str, optional): Prefijo para los nombres de archivo.
        
    Returns:
        dict: Diccionario con rutas a los archivos generados.
    """
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Generar informe CSV
    report_df = pd.DataFrame({
        'Feature': importances.index,
        'Importance': importances.values,
        'Selected': [feature in selected_features for feature in importances.index]
    }).sort_values('Importance', ascending=False)
    
    report_path = os.path.join(output_dir, f"{prefix}_report.csv")
    report_df.to_csv(report_path, index=False)
    
    # Generar visualizaciones
    plots = {}
    
    # 1. Gráfico de importancia de características
    importance_plot_path = os.path.join(output_dir, f"{prefix}_importance.png")
    plot_feature_importance(
        importances, selected_features, 
        save_path=importance_plot_path
    )
    plots['importance'] = importance_plot_path
    
    # 2. Matriz de correlación
    correlation_plot_path = os.path.join(output_dir, f"{prefix}_correlation.png")
    plot_correlation_matrix(
        X, selected_features, 
        save_path=correlation_plot_path
    )
    plots['correlation'] = correlation_plot_path
    
    # 3. Distribución de características seleccionadas
    if selected_features:
        distribution_plot_path = os.path.join(output_dir, f"{prefix}_distribution.png")
        plot_feature_distribution(
            X, selected_features, 
            save_path=distribution_plot_path
        )
        plots['distribution'] = distribution_plot_path
    
    # 4. Importancia por lag
    lag_plot_path = os.path.join(output_dir, f"{prefix}_lag_importance.png")
    try:
        plot_lag_importance(
            importances, 
            save_path=lag_plot_path
        )
        plots['lag_importance'] = lag_plot_path
    except:
        # Puede fallar si no hay patrón de lag en las características
        pass
    
    # 5. Coordenadas paralelas
    if len(selected_features) <= 20:  # Solo si no hay demasiadas características
        parallel_plot_path = os.path.join(output_dir, f"{prefix}_parallel.png")
        plot_parallel_coordinates(
            X, selected_features, 
            save_path=parallel_plot_path
        )
        plots['parallel'] = parallel_plot_path
    
    return {
        'report': report_path,
        'plots': plots
    }
