#!/usr/bin/env python3
"""
Métodos de selección de atributos basados en modelos para series temporales.

Este módulo implementa métodos de selección de atributos basados en modelos
para series temporales multivariables, incluyendo:
- Feature importance de Random Forest
- Lasso (L1)
- Elastic Net
- Recursive Feature Elimination (RFE)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, ElasticNetCV

# Importar la clase base desde el módulo de correlación
from .correlation_based import BaseFeatureSelector


class RandomForestSelector(BaseFeatureSelector):
    """Selector de características basado en importancia de Random Forest."""
    
    def __init__(self, n_features=None, threshold=None, n_estimators=100, 
                 max_depth=None, random_state=42, verbose=False):
        """
        Inicializa el selector basado en Random Forest.
        
        Args:
            n_features (int, optional): Número de características a seleccionar.
            threshold (float, optional): Umbral de importancia para seleccionar características.
            n_estimators (int, optional): Número de árboles en el bosque.
            max_depth (int, optional): Profundidad máxima de los árboles.
            random_state (int, optional): Semilla para reproducibilidad.
            verbose (bool, optional): Si es True, muestra información detallada.
        """
        super().__init__(n_features, threshold, verbose)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
    
    def fit(self, X, y=None):
        """
        Ajusta el selector entrenando un modelo Random Forest y extrayendo importancias.
        
        Args:
            X (DataFrame): Datos de entrada.
            y (Series): Variable objetivo.
            
        Returns:
            self: El selector ajustado.
        """
        if y is None:
            raise ValueError("La variable objetivo (y) es requerida para este selector")
        
        if self.verbose:
            print(f"Entrenando Random Forest con {self.n_estimators} árboles...")
        
        # Filtrar solo columnas numéricas
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_numeric = X[numeric_cols]
        
        # Crear y entrenar el modelo
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1 if self.verbose else 0
        )
        self.model.fit(X_numeric, y)
        
        # Extraer importancias
        self.feature_importances_ = pd.Series(
            self.model.feature_importances_,
            index=numeric_cols
        )
        
        # Seleccionar características
        self.selected_features_ = self._select_features()
        
        if self.verbose:
            print(f"Seleccionadas {len(self.selected_features_)} características basadas en Random Forest")
        
        return self
    
    def plot_feature_importances(self, top_n=None, figsize=(10, 8), save_path=None):
        """
        Visualiza las importancias de las características del Random Forest.
        
        Args:
            top_n (int, optional): Número de características principales a mostrar.
            figsize (tuple, optional): Tamaño de la figura.
            save_path (str, optional): Ruta para guardar la figura.
            
        Returns:
            matplotlib.figure.Figure: La figura generada.
        """
        return super().plot_feature_importances(top_n, figsize, save_path)


class LassoSelector(BaseFeatureSelector):
    """Selector de características basado en Lasso (L1)."""
    
    def __init__(self, n_features=None, threshold=None, alpha=1.0, 
                 max_iter=1000, random_state=42, verbose=False):
        """
        Inicializa el selector basado en Lasso.
        
        Args:
            n_features (int, optional): Número de características a seleccionar.
            threshold (float, optional): Umbral de coeficiente para seleccionar características.
            alpha (float, optional): Parámetro de regularización.
            max_iter (int, optional): Número máximo de iteraciones.
            random_state (int, optional): Semilla para reproducibilidad.
            verbose (bool, optional): Si es True, muestra información detallada.
        """
        super().__init__(n_features, threshold, verbose)
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        """
        Ajusta el selector entrenando un modelo Lasso y extrayendo coeficientes.
        
        Args:
            X (DataFrame): Datos de entrada.
            y (Series): Variable objetivo.
            
        Returns:
            self: El selector ajustado.
        """
        if y is None:
            raise ValueError("La variable objetivo (y) es requerida para este selector")
        
        if self.verbose:
            print(f"Entrenando Lasso con alpha={self.alpha}...")
        
        # Filtrar solo columnas numéricas
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_numeric = X[numeric_cols]
        
        # Escalar datos
        X_scaled = self.scaler.fit_transform(X_numeric)
        
        # Crear y entrenar el modelo
        self.model = LassoCV(
            cv=5, 
            random_state=self.random_state, 
            max_iter=self.max_iter
        )
        self.model.fit(X_scaled, y)
        if self.verbose:
            print(f"[INFO] Alpha optimizado por LassoCV: {self.model.alpha_:.6f}")
        # Extraer coeficientes (en valor absoluto para importancia)
        self.feature_importances_ = pd.Series(
            np.abs(self.model.coef_),
            index=numeric_cols
        )
        
        # Seleccionar características
        self.selected_features_ = self._select_features()
        
        if self.verbose:
            print(f"Seleccionadas {len(self.selected_features_)} características basadas en Lasso")
        
        return self
    
    def get_coefficients(self):
        """
        Devuelve los coeficientes del modelo Lasso.
        
        Returns:
            Series: Coeficientes del modelo.
        """
        if self.model is None:
            raise ValueError("El selector debe ser ajustado antes de obtener coeficientes")
        return pd.Series(self.model.coef_, index=self.feature_importances_.index)


class ElasticNetSelector(BaseFeatureSelector):
    """Selector de características basado en Elastic Net."""
    
    def __init__(self, n_features=None, threshold=None, alpha=1.0, l1_ratio=0.5,
                 max_iter=1000, random_state=42, verbose=False):
        """
        Inicializa el selector basado en Elastic Net.
        
        Args:
            n_features (int, optional): Número de características a seleccionar.
            threshold (float, optional): Umbral de coeficiente para seleccionar características.
            alpha (float, optional): Parámetro de regularización.
            l1_ratio (float, optional): Ratio de mezcla entre L1 y L2 (1 = Lasso, 0 = Ridge).
            max_iter (int, optional): Número máximo de iteraciones.
            random_state (int, optional): Semilla para reproducibilidad.
            verbose (bool, optional): Si es True, muestra información detallada.
        """
        super().__init__(n_features, threshold, verbose)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        """
        Ajusta el selector entrenando un modelo Elastic Net y extrayendo coeficientes.
        
        Args:
            X (DataFrame): Datos de entrada.
            y (Series): Variable objetivo.
            
        Returns:
            self: El selector ajustado.
        """
        if y is None:
            raise ValueError("La variable objetivo (y) es requerida para este selector")
        
        if self.verbose:
            print(f"Entrenando Elastic Net con alpha={self.alpha}, l1_ratio={self.l1_ratio}...")
        
        # Filtrar solo columnas numéricas
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_numeric = X[numeric_cols]
        
        # Escalar datos
        X_scaled = self.scaler.fit_transform(X_numeric)
        
        # Crear y entrenar el modelo
        self.model = ElasticNetCV(
            cv=5, 
            l1_ratio=[.1, .5, .7, .9, .95, .99, 1], # Prueba diferentes mezclas L1/L2
            random_state=self.random_state, 
            max_iter=self.max_iter
        )
        self.model.fit(X_scaled, y)
        if self.verbose:
            print(f"[INFO] Alpha optimizado por ElasticNetCV: {self.model.alpha_:.6f}")
        # Extraer coeficientes (en valor absoluto para importancia)
        self.feature_importances_ = pd.Series(
            np.abs(self.model.coef_),
            index=numeric_cols
        )
        
        # Seleccionar características
        self.selected_features_ = self._select_features()
        
        if self.verbose:
            print(f"Seleccionadas {len(self.selected_features_)} características basadas en Elastic Net")
        
        return self
    
    def get_coefficients(self):
        """
        Devuelve los coeficientes del modelo Elastic Net.
        
        Returns:
            Series: Coeficientes del modelo.
        """
        if self.model is None:
            raise ValueError("El selector debe ser ajustado antes de obtener coeficientes")
        return pd.Series(self.model.coef_, index=self.feature_importances_.index)


class RFESelector(BaseFeatureSelector):
    """Selector de características basado en Recursive Feature Elimination (RFE)."""
    
    def __init__(self, n_features=None, step=1, estimator=None, verbose=False):
        """
        Inicializa el selector basado en RFE.
        
        Args:
            n_features (int, optional): Número de características a seleccionar.
            step (int, optional): Número de características a eliminar en cada iteración.
            estimator (object, optional): Estimador base para RFE.
            verbose (bool, optional): Si es True, muestra información detallada.
        """
        super().__init__(n_features, None, verbose)  # RFE no usa threshold
        self.step = step
        self.estimator = estimator or RandomForestRegressor(n_estimators=100, random_state=42)
        self.rfe = None
    
    def fit(self, X, y=None):
        """
        Ajusta el selector usando RFE.
        
        Args:
            X (DataFrame): Datos de entrada.
            y (Series): Variable objetivo.
            
        Returns:
            self: El selector ajustado.
        """
        if y is None:
            raise ValueError("La variable objetivo (y) es requerida para este selector")
        
        if self.n_features is None:
            raise ValueError("El número de características (n_features) es requerido para RFE")
        
        if self.verbose:
            print(f"Ejecutando RFE para seleccionar {self.n_features} características...")
        
        # Filtrar solo columnas numéricas
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_numeric = X[numeric_cols]
        
        # Crear y ajustar RFE
        self.rfe = RFE(
            estimator=self.estimator,
            n_features_to_select=self.n_features,
            step=self.step,
            verbose=1 if self.verbose else 0
        )
        self.rfe.fit(X_numeric, y)
        
        # Extraer importancias (1 para seleccionadas, 0 para no seleccionadas)
        self.feature_importances_ = pd.Series(
            self.rfe.ranking_,
            index=numeric_cols
        )
        # Invertir ranking para que valores más altos sean más importantes
        self.feature_importances_ = 1 / self.feature_importances_
        
        # Seleccionar características
        self.selected_features_ = [
            feature for feature, selected in zip(numeric_cols, self.rfe.support_)
            if selected
        ]
        
        if self.verbose:
            print(f"Seleccionadas {len(self.selected_features_)} características usando RFE")
        
        return self


def create_model_selector(method='random_forest', n_features=None, threshold=None, **kwargs):
    """
    Crea un selector de características basado en modelos según el método especificado.
    
    Args:
        method (str): Método de selección ('random_forest', 'lasso', 'elastic_net', 'rfe').
        n_features (int, optional): Número de características a seleccionar.
        threshold (float, optional): Umbral para la selección de características.
        **kwargs: Argumentos adicionales específicos para cada método.
        
    Returns:
        BaseFeatureSelector: El selector de características correspondiente.
    """
    verbose = kwargs.get('verbose', False)
    
    if method == 'random_forest':
        return RandomForestSelector(
            n_features=n_features,
            threshold=threshold,
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            random_state=kwargs.get('random_state', 42),
            verbose=verbose
        )
    elif method == 'lasso':
        return LassoSelector(
            n_features=n_features,
            threshold=threshold,
            alpha=kwargs.get('alpha', 1.0),
            max_iter=kwargs.get('max_iter', 1000),
            random_state=kwargs.get('random_state', 42),
            verbose=verbose
        )
    elif method == 'elastic_net':
        return ElasticNetSelector(
            n_features=n_features,
            threshold=threshold,
            alpha=kwargs.get('alpha', 1.0),
            l1_ratio=kwargs.get('l1_ratio', 0.5),
            max_iter=kwargs.get('max_iter', 1000),
            random_state=kwargs.get('random_state', 42),
            verbose=verbose
        )
    elif method == 'rfe':
        return RFESelector(
            n_features=n_features,
            step=kwargs.get('step', 1),
            estimator=kwargs.get('estimator', None),
            verbose=verbose
        )
    else:
        raise ValueError(f"Método desconocido: {method}. Opciones válidas: 'random_forest', 'lasso', 'elastic_net', 'rfe'")
