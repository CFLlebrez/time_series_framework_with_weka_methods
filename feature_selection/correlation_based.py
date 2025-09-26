#!/usr/bin/env python3
"""
Métodos de selección de atributos basados en correlación para series temporales.

Este módulo implementa métodos de selección de atributos basados en correlación
para series temporales multivariables, incluyendo:
- Correlación de Pearson
- Análisis de correlación cruzada (CCF)
- Información mutua
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class BaseFeatureSelector:
    """Clase base para todos los métodos de selección de atributos."""
    
    def __init__(self, n_features=None, threshold=None, verbose=False):
        """
        Inicializa el selector de características.
        
        Args:
            n_features (int, optional): Número de características a seleccionar.
            threshold (float, optional): Umbral para la selección de características.
            verbose (bool, optional): Si es True, muestra información detallada.
        """
        self.n_features = n_features
        self.threshold = threshold
        self.verbose = verbose
        self.feature_importances_ = None
        self.selected_features_ = None
        
    def fit(self, X, y=None):
        """
        Ajusta el selector a los datos.
        
        Args:
            X (DataFrame): Datos de entrada.
            y (Series, optional): Variable objetivo.
            
        Returns:
            self: El selector ajustado.
        """
        raise NotImplementedError("Las subclases deben implementar este método")
        
    def transform(self, X):
        """
        Transforma los datos usando solo las características seleccionadas.
        
        Args:
            X (DataFrame): Datos de entrada.
            
        Returns:
            DataFrame: Datos transformados con solo las características seleccionadas.
        """
        if self.selected_features_ is None:
            raise ValueError("El selector debe ser ajustado antes de transformar")
        return X[self.selected_features_]
        
    def fit_transform(self, X, y=None):
        """
        Ajusta el selector y transforma los datos.
        
        Args:
            X (DataFrame): Datos de entrada.
            y (Series, optional): Variable objetivo.
            
        Returns:
            DataFrame: Datos transformados con solo las características seleccionadas.
        """
        self.fit(X, y)
        return self.transform(X)
        
    def get_feature_importances(self):
        """
        Devuelve las importancias de las características.
        
        Returns:
            Series: Importancias de las características.
        """
        if self.feature_importances_ is None:
            raise ValueError("El selector debe ser ajustado antes de obtener importancias")
        return self.feature_importances_
        
    def get_selected_features(self):
        """
        Devuelve las características seleccionadas.
        
        Returns:
            list: Nombres de las características seleccionadas.
        """
        if self.selected_features_ is None:
            raise ValueError("El selector debe ser ajustado antes de obtener características")
        return self.selected_features_
    
    def _select_features(self):
        """
        Selecciona las características basadas en sus importancias.
        
        Returns:
            list: Nombres de las características seleccionadas.
        """
        if self.feature_importances_ is None:
            raise ValueError("Las importancias de características deben calcularse antes de seleccionar")
        
        # Ordenar características por importancia
        sorted_features = self.feature_importances_.sort_values(ascending=False)
        
        # Seleccionar características basadas en n_features o threshold
        if self.n_features is not None:
            selected = sorted_features.iloc[:self.n_features].index.tolist()
        elif self.threshold is not None:
            selected = sorted_features[sorted_features >= self.threshold].index.tolist()
        else:
            # Por defecto, seleccionar características con importancia positiva
            selected = sorted_features[sorted_features > 0].index.tolist()
        
        return selected
    
    def plot_feature_importances(self, top_n=None, figsize=(10, 8), save_path=None):
        """
        Visualiza las importancias de las características.
        
        Args:
            top_n (int, optional): Número de características principales a mostrar.
            figsize (tuple, optional): Tamaño de la figura.
            save_path (str, optional): Ruta para guardar la figura.
            
        Returns:
            matplotlib.figure.Figure: La figura generada.
        """
        if self.feature_importances_ is None:
            raise ValueError("El selector debe ser ajustado antes de visualizar importancias")
        
        # Ordenar características por importancia
        sorted_importances = self.feature_importances_.sort_values(ascending=False)
        
        # Limitar a top_n si se especifica
        if top_n is not None:
            sorted_importances = sorted_importances.iloc[:top_n]
        
        # Crear figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Crear gráfico de barras
        sorted_importances.plot(kind='bar', ax=ax)
        
        # Añadir títulos y etiquetas
        ax.set_title('Importancia de las Características')
        ax.set_xlabel('Características')
        ax.set_ylabel('Importancia')
        
        # Rotar etiquetas del eje x para mejor legibilidad
        plt.xticks(rotation=90)
        
        # Ajustar diseño
        plt.tight_layout()
        
        # Guardar figura si se especifica ruta
        if save_path:
            plt.savefig(save_path)
        
        return fig


class PearsonCorrelationSelector(BaseFeatureSelector):
    """Selector de características basado en correlación de Pearson."""
    
    def __init__(self, n_features=None, threshold=None, absolute=True, verbose=False):
        """
        Inicializa el selector basado en correlación de Pearson.
        
        Args:
            n_features (int, optional): Número de características a seleccionar.
            threshold (float, optional): Umbral de correlación para seleccionar características.
            absolute (bool, optional): Si es True, usa el valor absoluto de la correlación.
            verbose (bool, optional): Si es True, muestra información detallada.
        """
        super().__init__(n_features, threshold, verbose)
        self.absolute = absolute
    
    def fit(self, X, y=None):
        """
        Ajusta el selector calculando correlaciones de Pearson con la variable objetivo.
        
        Args:
            X (DataFrame): Datos de entrada.
            y (Series): Variable objetivo.
            
        Returns:
            self: El selector ajustado.
        """
        if y is None:
            raise ValueError("La variable objetivo (y) es requerida para este selector")
        
        if self.verbose:
            print("Calculando correlaciones de Pearson...")
        
        # Asegurar que y es una Serie de pandas
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Calcular correlación de cada característica con la variable objetivo
        correlations = {}
        for col in X.columns:
            if X[col].dtype.kind in 'bifc':  # Verificar si la columna es numérica
                corr = stats.pearsonr(X[col], y)[0]
                correlations[col] = abs(corr) if self.absolute else corr
        
        # Convertir a Series
        self.feature_importances_ = pd.Series(correlations)
        
        # Seleccionar características
        self.selected_features_ = self._select_features()
        
        if self.verbose:
            print(f"Seleccionadas {len(self.selected_features_)} características basadas en correlación de Pearson")
        
        return self


class CrossCorrelationSelector(BaseFeatureSelector):
    """Selector de características basado en análisis de correlación cruzada (CCF)."""
    
    def __init__(self, n_features=None, threshold=None, max_lag=10, absolute=True, verbose=False):
        """
        Inicializa el selector basado en correlación cruzada.
        
        Args:
            n_features (int, optional): Número de características a seleccionar.
            threshold (float, optional): Umbral de correlación para seleccionar características.
            max_lag (int, optional): Máximo lag a considerar.
            absolute (bool, optional): Si es True, usa el valor absoluto de la correlación.
            verbose (bool, optional): Si es True, muestra información detallada.
        """
        super().__init__(n_features, threshold, verbose)
        self.max_lag = max_lag
        self.absolute = absolute
        self.best_lags_ = None
    
    def fit(self, X, y=None):
        """
        Ajusta el selector calculando correlaciones cruzadas con la variable objetivo.
        
        Args:
            X (DataFrame): Datos de entrada.
            y (Series): Variable objetivo.
            
        Returns:
            self: El selector ajustado.
        """
        if y is None:
            raise ValueError("La variable objetivo (y) es requerida para este selector")
        
        if self.verbose:
            print(f"Calculando correlaciones cruzadas con máximo lag {self.max_lag}...")
        
        # Asegurar que y es una Serie de pandas
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Calcular correlación cruzada para cada característica y lag
        correlations = {}
        best_lags = {}
        
        for col in tqdm(X.columns, disable=not self.verbose):
            if X[col].dtype.kind in 'bifc':  # Verificar si la columna es numérica
                max_corr = 0
                best_lag = 0
                
                # Probar diferentes lags
                for lag in range(self.max_lag + 1):
                    # Alinear series con el lag
                    x_lagged = X[col].shift(lag)
                    # Eliminar NaNs
                    valid_idx = ~x_lagged.isna() #~ works as ! for pandas
                    if valid_idx.sum() > 10:  # Asegurar suficientes datos
                        corr = stats.pearsonr(x_lagged[valid_idx], y[valid_idx])[0]
                        corr_abs = abs(corr)
                        if corr_abs > max_corr:
                            max_corr = corr_abs
                            best_lag = lag
                            best_corr = corr
                
                # Guardar la mejor correlación y lag
                correlations[col] = abs(best_corr) if self.absolute else best_corr
                best_lags[col] = best_lag
        
        # Convertir a Series
        self.feature_importances_ = pd.Series(correlations)
        self.best_lags_ = pd.Series(best_lags)
        
        # Seleccionar características
        self.selected_features_ = self._select_features()
        
        if self.verbose:
            print(f"Seleccionadas {len(self.selected_features_)} características basadas en correlación cruzada")
        
        return self
    
    def get_best_lags(self):
        """
        Devuelve los mejores lags para cada característica.
        
        Returns:
            Series: Mejores lags para cada característica.
        """
        if self.best_lags_ is None:
            raise ValueError("El selector debe ser ajustado antes de obtener los mejores lags")
        return self.best_lags_
    
    def plot_ccf_heatmap(self, X, y, top_n=10, figsize=(12, 8), save_path=None):
        """
        Visualiza un mapa de calor de correlaciones cruzadas para las principales características.
        
        Args:
            X (DataFrame): Datos de entrada.
            y (Series): Variable objetivo.
            top_n (int, optional): Número de características principales a mostrar.
            figsize (tuple, optional): Tamaño de la figura.
            save_path (str, optional): Ruta para guardar la figura.
            
        Returns:
            matplotlib.figure.Figure: La figura generada.
        """
        if self.feature_importances_ is None:
            raise ValueError("El selector debe ser ajustado antes de visualizar")
        
        # Seleccionar top_n características
        top_features = self.feature_importances_.sort_values(ascending=False).iloc[:top_n].index
        
        # Calcular matriz de correlación cruzada
        ccf_matrix = np.zeros((len(top_features), self.max_lag + 1))
        
        for i, feature in enumerate(top_features): # (i,lag) works as (i,j)
            for lag in range(self.max_lag + 1):
                x_lagged = X[feature].shift(lag)
                valid_idx = ~x_lagged.isna()
                if valid_idx.sum() > 10:
                    corr = stats.pearsonr(x_lagged[valid_idx], y[valid_idx])[0]
                    ccf_matrix[i, lag] = corr
        
        # Crear figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Crear mapa de calor
        sns.heatmap(ccf_matrix, 
                   xticklabels=range(self.max_lag + 1),
                   yticklabels=top_features,
                   cmap='coolwarm',
                   center=0,
                   ax=ax)
        
        # Añadir títulos y etiquetas
        ax.set_title('Correlación Cruzada por Lag')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Característica')
        
        # Ajustar diseño
        plt.tight_layout()
        
        # Guardar figura si se especifica ruta
        if save_path:
            plt.savefig(save_path)
        
        return fig


class MutualInformationSelector(BaseFeatureSelector):
    """Selector de características basado en información mutua."""
    
    def __init__(self, n_features=None, threshold=None, n_neighbors=3, verbose=False):
        """
        Inicializa el selector basado en información mutua.
        
        Args:
            n_features (int, optional): Número de características a seleccionar.
            threshold (float, optional): Umbral de información mutua para seleccionar características.
            n_neighbors (int, optional): Número de vecinos para estimar la información mutua.
            verbose (bool, optional): Si es True, muestra información detallada.
        """
        super().__init__(n_features, threshold, verbose)
        self.n_neighbors = n_neighbors
    
    def fit(self, X, y=None):
        """
        Ajusta el selector calculando información mutua con la variable objetivo.
        
        Args:
            X (DataFrame): Datos de entrada.
            y (Series): Variable objetivo.
            
        Returns:
            self: El selector ajustado.
        """
        if y is None:
            raise ValueError("La variable objetivo (y) es requerida para este selector")
        
        if self.verbose:
            print("Calculando información mutua...")
        
        # Filtrar solo columnas numéricas
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_numeric = X[numeric_cols]
        
        # Calcular información mutua
        mi_values = mutual_info_regression(X_numeric, y, n_neighbors=self.n_neighbors)
        
        # Convertir a Series
        self.feature_importances_ = pd.Series(mi_values, index=numeric_cols)
        
        # Seleccionar características
        self.selected_features_ = self._select_features()
        
        if self.verbose:
            print(f"Seleccionadas {len(self.selected_features_)} características basadas en información mutua")
        
        return self


def create_correlation_selector(method='pearson', n_features=None, threshold=None, max_lag=10, absolute=True, verbose=False):
    """
    Crea un selector de características basado en correlación según el método especificado.
    
    Args:
        method (str): Método de selección ('pearson', 'ccf', 'mutual_info').
        n_features (int, optional): Número de características a seleccionar.
        threshold (float, optional): Umbral para la selección de características.
        max_lag (int, optional): Máximo lag a considerar para CCF.
        absolute (bool, optional): Si es True, usa el valor absoluto de la correlación.
        verbose (bool, optional): Si es True, muestra información detallada.
        
    Returns:
        BaseFeatureSelector: El selector de características correspondiente.
    """
    if method == 'pearson':
        return PearsonCorrelationSelector(n_features, threshold, absolute, verbose)
    elif method == 'ccf':
        return CrossCorrelationSelector(n_features, threshold, max_lag, absolute, verbose)
    elif method == 'mutual_info':
        return MutualInformationSelector(n_features, threshold, verbose=verbose)
    else:
        raise ValueError(f"Método desconocido: {method}. Opciones válidas: 'pearson', 'ccf', 'mutual_info'")
