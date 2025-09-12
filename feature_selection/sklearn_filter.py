#!/usr/bin/env python3
"""
Métodos de selección de atributos basados en filtros de scikit-learn para series temporales.

Este módulo implementa métodos de selección de atributos basados en filtros de scikit-learn,
incluyendo:
- SelectKBest con diferentes funciones de puntuación
- SelectPercentile
- GenericUnivariateSelect
- VarianceThreshold
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, GenericUnivariateSelect,
    VarianceThreshold, f_regression, mutual_info_regression
)
from tqdm import tqdm

# Importar la clase base desde el módulo de correlación
from .correlation_based import BaseFeatureSelector


class SKLearnFilterSelector(BaseFeatureSelector):
    """Selector de características basado en métodos de filtro de scikit-learn."""
    
    def __init__(self, method='selectkbest', n_features=None, percentile=None, 
                 score_func=f_regression, strategy='k_best', param=None, 
                 threshold=0.0, verbose=False):
        """
        Inicializa el selector basado en filtros de scikit-learn.
        
        Args:
            method (str): Método de filtro ('selectkbest', 'selectpercentile', 
                         'genericunivariateselect', 'variancethreshold').
            n_features (int, optional): Número de características a seleccionar (para SelectKBest).
            percentile (int, optional): Percentil de características a seleccionar (para SelectPercentile).
            score_func (callable, optional): Función de puntuación para métodos univariados.
            strategy (str, optional): Estrategia para GenericUnivariateSelect.
            param (float, optional): Parámetro para la estrategia de GenericUnivariateSelect.
            threshold (float, optional): Umbral para VarianceThreshold.
            verbose (bool, optional): Si es True, muestra información detallada.
        """
        super().__init__(n_features, None, verbose)
        self.method = method
        self.percentile = percentile
        self.score_func = score_func
        self.strategy = strategy
        self.param = param
        self.threshold = threshold
        self.selector = None
        
    def fit(self, X, y=None):
        """
        Ajusta el selector a los datos.
        
        Args:
            X (DataFrame): Datos de entrada.
            y (Series, optional): Variable objetivo.
            
        Returns:
            self: El selector ajustado.
        """
        if self.method in ['selectkbest', 'selectpercentile', 'genericunivariateselect'] and y is None:
            raise ValueError(f"La variable objetivo (y) es requerida para el método {self.method}")
        
        # Filtrar solo columnas numéricas
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_numeric = X[numeric_cols]
        
        if self.verbose:
            print(f"Aplicando método de filtro {self.method} de scikit-learn...")
        
        # Crear selector según el método
        if self.method == 'selectkbest':
            if self.n_features is None:
                raise ValueError("El número de características (n_features) es requerido para SelectKBest")
            self.selector = SelectKBest(score_func=self.score_func, k=self.n_features)
            
        elif self.method == 'selectpercentile':
            if self.percentile is None:
                raise ValueError("El percentil (percentile) es requerido para SelectPercentile")
            self.selector = SelectPercentile(score_func=self.score_func, percentile=self.percentile)
            
        elif self.method == 'genericunivariateselect':
            self.selector = GenericUnivariateSelect(
                score_func=self.score_func, 
                mode=self.strategy,
                param=self.param if self.param is not None else (
                    self.n_features if self.strategy == 'k_best' else 
                    self.percentile if self.strategy == 'percentile' else None
                )
            )
            
        elif self.method == 'variancethreshold':
            self.selector = VarianceThreshold(threshold=self.threshold)
            
        else:
            raise ValueError(f"Método desconocido: {self.method}")
        
        # Ajustar selector
        if self.method == 'variancethreshold':
            self.selector.fit(X_numeric)
        else:
            self.selector.fit(X_numeric, y)
        
        # Obtener características seleccionadas
        support = self.selector.get_support()
        self.selected_features_ = [feature for feature, selected in zip(numeric_cols, support) if selected]
        
        # Obtener puntuaciones (si están disponibles)
        if hasattr(self.selector, 'scores_'):
            self.feature_importances_ = pd.Series(
                self.selector.scores_,
                index=numeric_cols
            )
        else:
            # Para VarianceThreshold, usar varianzas como importancias
            if self.method == 'variancethreshold':
                variances = X_numeric.var()
                self.feature_importances_ = variances
            else:
                # Crear importancias binarias (1 para seleccionadas, 0 para no seleccionadas)
                self.feature_importances_ = pd.Series(
                    [1.0 if s else 0.0 for s in support],
                    index=numeric_cols
                )
        
        if self.verbose:
            print(f"Seleccionadas {len(self.selected_features_)} características usando {self.method}")
        
        return self
    
    def transform(self, X):
        """
        Transforma los datos usando solo las características seleccionadas.
        
        Args:
            X (DataFrame): Datos de entrada.
            
        Returns:
            DataFrame: Datos transformados con solo las características seleccionadas.
        """
        if self.selector is None:
            raise ValueError("El selector debe ser ajustado antes de transformar")
        
        # Filtrar solo columnas numéricas
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_numeric = X[numeric_cols]
        
        # Aplicar transformación
        X_transformed = self.selector.transform(X_numeric)
        
        # Convertir a DataFrame
        X_transformed_df = pd.DataFrame(
            X_transformed,
            columns=self.selected_features_,
            index=X.index
        )
        
        return X_transformed_df


def create_sklearn_filter_selector(method='selectkbest', n_features=None, percentile=None, 
                                  score_func_name='f_regression', strategy='k_best', 
                                  param=None, threshold=0.0, verbose=False):
    """
    Crea un selector de características basado en filtros de scikit-learn.
    
    Args:
        method (str): Método de filtro ('selectkbest', 'selectpercentile', 
                     'genericunivariateselect', 'variancethreshold').
        n_features (int, optional): Número de características a seleccionar.
        percentile (int, optional): Percentil de características a seleccionar.
        score_func_name (str, optional): Nombre de la función de puntuación.
        strategy (str, optional): Estrategia para GenericUnivariateSelect.
        param (float, optional): Parámetro para la estrategia de GenericUnivariateSelect.
        threshold (float, optional): Umbral para VarianceThreshold.
        verbose (bool, optional): Si es True, muestra información detallada.
        
    Returns:
        SKLearnFilterSelector: El selector de características correspondiente.
    """
    # Determinar función de puntuación
    if score_func_name == 'f_regression':
        score_func = f_regression
    elif score_func_name == 'mutual_info_regression':
        score_func = mutual_info_regression
    else:
        raise ValueError(f"Función de puntuación desconocida: {score_func_name}")
    
    return SKLearnFilterSelector(
        method=method,
        n_features=n_features,
        percentile=percentile,
        score_func=score_func,
        strategy=strategy,
        param=param,
        threshold=threshold,
        verbose=verbose
    )
