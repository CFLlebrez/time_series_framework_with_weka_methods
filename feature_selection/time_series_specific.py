#!/usr/bin/env python3
"""
Métodos de selección de atributos específicos para series temporales.

Este módulo implementa métodos de selección de atributos específicos para series temporales,
incluyendo:
- Análisis de causalidad de Granger
- Análisis de componentes principales (PCA)
- Análisis espectral
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests, acf, pacf
from sklearn.decomposition import PCA
from scipy.signal import periodogram
from tqdm import tqdm

# Importar la clase base desde el módulo de correlación
from .correlation_based import BaseFeatureSelector


class GrangerCausalitySelector(BaseFeatureSelector):
    """Selector de características basado en causalidad de Granger."""
    
    def __init__(self, n_features=None, threshold=None, max_lag=5, test='ssr_chi2test', verbose=False):
        """
        Inicializa el selector basado en causalidad de Granger.
        
        Args:
            n_features (int, optional): Número de características a seleccionar.
            threshold (float, optional): Umbral de p-valor para seleccionar características.
            max_lag (int, optional): Máximo lag a considerar.
            test (str, optional): Test estadístico a usar ('ssr_chi2test', 'ssr_ftest', 'lrtest', 'params_ftest').
            verbose (bool, optional): Si es True, muestra información detallada.
        """
        super().__init__(n_features, threshold, verbose)
        self.max_lag = 1
        self.test = test
        self.p_values_ = None
        self.best_lags_ = None
    
    def fit(self, X, y=None):
        """
        Ajusta el selector calculando causalidad de Granger con la variable objetivo.
        
        Args:
            X (DataFrame): Datos de entrada.
            y (Series): Variable objetivo.
            
        Returns:
            self: El selector ajustado.
        """
        if y is None:
            raise ValueError("La variable objetivo (y) es requerida para este selector")
        
        if self.verbose:
            print(f"Calculando causalidad de Granger con máximo lag {self.max_lag}...")
        
        # Asegurar que y es una Serie de pandas
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Filtrar solo columnas numéricas
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        
        # Calcular causalidad de Granger para cada característica
        p_values = {}
        best_lags = {}
        
        for col in tqdm(numeric_cols, disable=not self.verbose):
            # Crear DataFrame con la variable objetivo y la característica actual
            df_test = pd.DataFrame({
                'y': y,
                'x': X[col]
            }).dropna()
            
            if len(df_test) <= self.max_lag + 1:
                if self.verbose:
                    print(f"Omitiendo {col}: datos insuficientes después de eliminar NaNs")
                continue
            
            try:
                # Probar causalidad de Granger (x -> y)
                gc_res = grangercausalitytests(df_test[['y', 'x']], maxlag=self.max_lag, verbose=False)
                
                # Extraer p-valores para cada lag
                p_vals = [gc_res[lag+1][0][self.test][1] for lag in range(self.max_lag)]
                
                # Encontrar el mejor lag (menor p-valor)
                min_p_val = min(p_vals)
                best_lag = p_vals.index(min_p_val) + 1
                
                # Guardar resultados
                p_values[col] = min_p_val
                best_lags[col] = best_lag
                
            except Exception as e:
                if self.verbose:
                    print(f"Error al calcular causalidad de Granger para {col}: {e}")
        
        # Convertir a Series
        self.p_values_ = pd.Series(p_values)
        self.best_lags_ = pd.Series(best_lags)
        
        # Calcular importancias como 1 - p_valor (mayor importancia para menor p-valor)
        self.feature_importances_ = 1 - self.p_values_
        
        # Seleccionar características
        self.selected_features_ = self._select_features()
        
        if self.verbose:
            print(f"Seleccionadas {len(self.selected_features_)} características basadas en causalidad de Granger")
        
        return self
    
    def get_p_values(self):
        """
        Devuelve los p-valores para cada característica.
        
        Returns:
            Series: P-valores para cada característica.
        """
        if self.p_values_ is None:
            raise ValueError("El selector debe ser ajustado antes de obtener p-valores")
        return self.p_values_
    
    def get_best_lags(self):
        """
        Devuelve los mejores lags para cada característica.
        
        Returns:
            Series: Mejores lags para cada característica.
        """
        if self.best_lags_ is None:
            raise ValueError("El selector debe ser ajustado antes de obtener los mejores lags")
        return self.best_lags_
    
    def plot_p_values(self, top_n=None, figsize=(10, 8), save_path=None):
        """
        Visualiza los p-valores para las principales características.
        
        Args:
            top_n (int, optional): Número de características principales a mostrar.
            figsize (tuple, optional): Tamaño de la figura.
            save_path (str, optional): Ruta para guardar la figura.
            
        Returns:
            matplotlib.figure.Figure: La figura generada.
        """
        if self.p_values_ is None:
            raise ValueError("El selector debe ser ajustado antes de visualizar p-valores")
        
        # Ordenar características por p-valor (ascendente)
        sorted_p_values = self.p_values_.sort_values()
        
        # Limitar a top_n si se especifica
        if top_n is not None:
            sorted_p_values = sorted_p_values.iloc[:top_n]
        
        # Crear figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Crear gráfico de barras
        sorted_p_values.plot(kind='bar', ax=ax)
        
        # Añadir línea horizontal en p=0.05
        ax.axhline(y=0.05, color='r', linestyle='--', label='p=0.05')
        
        # Añadir títulos y etiquetas
        ax.set_title('P-valores de Causalidad de Granger')
        ax.set_xlabel('Características')
        ax.set_ylabel('P-valor')
        ax.legend()
        
        # Rotar etiquetas del eje x para mejor legibilidad
        plt.xticks(rotation=90)
        
        # Ajustar diseño
        plt.tight_layout()
        
        # Guardar figura si se especifica ruta
        if save_path:
            plt.savefig(save_path)
        
        return fig


class PCASelector(BaseFeatureSelector):
    """Selector de características basado en Análisis de Componentes Principales (PCA)."""
    
    def __init__(self, n_components=None, variance_threshold=0.95, verbose=False):
        """
        Inicializa el selector basado en PCA.
        
        Args:
            n_components (int, optional): Número de componentes a seleccionar.
            variance_threshold (float, optional): Umbral de varianza explicada para seleccionar componentes.
            verbose (bool, optional): Si es True, muestra información detallada.
        """
        super().__init__(n_components, None, verbose)  # PCA no usa threshold en el sentido tradicional
        self.variance_threshold = variance_threshold
        self.pca = None
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.cumulative_variance_ = None
    
    def fit(self, X, y=None):
        """
        Ajusta el selector aplicando PCA a los datos.
        
        Args:
            X (DataFrame): Datos de entrada.
            y (Series, optional): Variable objetivo (no utilizada en PCA).
            
        Returns:
            self: El selector ajustado.
        """
        if self.verbose:
            print("Aplicando PCA...")
        
        # Filtrar solo columnas numéricas
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_numeric = X[numeric_cols]
        
        # Determinar número de componentes
        n_components = self.n_features
        if n_components is None:
            n_components = min(len(numeric_cols), len(X_numeric))
        
        # Crear y ajustar PCA
        self.pca = PCA(n_components=n_components)
        self.pca.fit(X_numeric)
        
        # Guardar componentes y varianza explicada
        self.components_ = pd.DataFrame(
            self.pca.components_,
            columns=numeric_cols,
            index=[f'PC{i+1}' for i in range(n_components)]
        )
        self.explained_variance_ratio_ = pd.Series(
            self.pca.explained_variance_ratio_,
            index=[f'PC{i+1}' for i in range(n_components)]
        )
        self.cumulative_variance_ = np.cumsum(self.pca.explained_variance_ratio_)
        
        # Calcular importancia de características basada en su contribución a los componentes principales
        feature_importances = {}
        for feature in numeric_cols:
            # Suma ponderada de los coeficientes de cada componente, ponderados por la varianza explicada
            importance = sum(abs(self.components_.loc[f'PC{i+1}', feature]) * self.explained_variance_ratio_.iloc[i] 
                             for i in range(n_components))
            feature_importances[feature] = importance
        
        self.feature_importances_ = pd.Series(feature_importances)
        
        # Determinar número de componentes a retener basado en varianza explicada
        if self.variance_threshold is not None:
            n_components_to_keep = np.argmax(self.cumulative_variance_ >= self.variance_threshold) + 1
        else:
            n_components_to_keep = n_components
        
        # Seleccionar características más importantes
        self.selected_features_ = self._select_features()
        
        if self.verbose:
            print(f"PCA completado. {n_components_to_keep} componentes explican {self.cumulative_variance_[n_components_to_keep-1]*100:.2f}% de la varianza.")
            print(f"Seleccionadas {len(self.selected_features_)} características basadas en su contribución a los componentes principales")
        
        return self
    
    def transform(self, X):
        """
        Transforma los datos usando PCA.
        
        Args:
            X (DataFrame): Datos de entrada.
            
        Returns:
            DataFrame: Datos transformados (componentes principales).
        """
        if self.pca is None:
            raise ValueError("El selector debe ser ajustado antes de transformar")
        
        # Filtrar solo columnas numéricas
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_numeric = X[numeric_cols]
        
        # Aplicar transformación PCA
        X_pca = self.pca.transform(X_numeric)
        
        # Convertir a DataFrame
        X_pca_df = pd.DataFrame(
            X_pca,
            columns=[f'PC{i+1}' for i in range(X_pca.shape[1])],
            index=X.index
        )
        
        return X_pca_df
    
    def plot_explained_variance(self, figsize=(10, 6), save_path=None):
        """
        Visualiza la varianza explicada por cada componente principal.
        
        Args:
            figsize (tuple, optional): Tamaño de la figura.
            save_path (str, optional): Ruta para guardar la figura.
            
        Returns:
            matplotlib.figure.Figure: La figura generada.
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("El selector debe ser ajustado antes de visualizar")
        
        # Crear figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Crear gráfico de barras para varianza individual
        ax.bar(
            range(len(self.explained_variance_ratio_)),
            self.explained_variance_ratio_,
            alpha=0.7,
            label='Varianza individual'
        )
        
        # Añadir línea para varianza acumulada
        ax.plot(
            range(len(self.cumulative_variance_)),
            self.cumulative_variance_,
            'o-',
            color='red',
            label='Varianza acumulada'
        )
        
        # Añadir línea horizontal en el umbral de varianza
        if self.variance_threshold is not None:
            ax.axhline(y=self.variance_threshold, color='g', linestyle='--', 
                      label=f'Umbral ({self.variance_threshold*100}%)')
        
        # Añadir títulos y etiquetas
        ax.set_title('Varianza Explicada por Componentes Principales')
        ax.set_xlabel('Componente Principal')
        ax.set_ylabel('Proporción de Varianza Explicada')
        ax.set_xticks(range(len(self.explained_variance_ratio_)))
        ax.set_xticklabels([f'PC{i+1}' for i in range(len(self.explained_variance_ratio_))])
        ax.legend()
        
        # Ajustar diseño
        plt.tight_layout()
        
        # Guardar figura si se especifica ruta
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def plot_component_heatmap(self, n_components=3, figsize=(12, 8), save_path=None):
        """
        Visualiza un mapa de calor de los coeficientes de los componentes principales.
        
        Args:
            n_components (int, optional): Número de componentes a mostrar.
            figsize (tuple, optional): Tamaño de la figura.
            save_path (str, optional): Ruta para guardar la figura.
            
        Returns:
            matplotlib.figure.Figure: La figura generada.
        """
        if self.components_ is None:
            raise ValueError("El selector debe ser ajustado antes de visualizar")
        
        # Limitar a n_components
        components_to_plot = self.components_.iloc[:n_components]
        
        # Crear figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Crear mapa de calor
        sns.heatmap(
            components_to_plot,
            cmap='coolwarm',
            center=0,
            ax=ax
        )
        
        # Añadir títulos y etiquetas
        ax.set_title('Coeficientes de los Componentes Principales')
        ax.set_ylabel('Componente Principal')
        ax.set_xlabel('Característica')
        
        # Ajustar diseño
        plt.tight_layout()
        
        # Guardar figura si se especifica ruta
        if save_path:
            plt.savefig(save_path)
        
        return fig


class SpectralSelector(BaseFeatureSelector):
    """Selector de características basado en análisis espectral."""
    
    def __init__(self, n_features=None, threshold=None, fs=1.0, method='periodogram', verbose=False):
        """
        Inicializa el selector basado en análisis espectral.
        
        Args:
            n_features (int, optional): Número de características a seleccionar.
            threshold (float, optional): Umbral de potencia espectral para seleccionar características.
            fs (float, optional): Frecuencia de muestreo.
            method (str, optional): Método de análisis espectral ('periodogram', 'acf').
            verbose (bool, optional): Si es True, muestra información detallada.
        """
        super().__init__(n_features, threshold, verbose)
        self.fs = fs
        self.method = method
        self.spectral_power_ = None
        self.frequencies_ = None
        self.acf_values_ = None
    
    def fit(self, X, y=None):
        """
        Ajusta el selector calculando características espectrales.
        
        Args:
            X (DataFrame): Datos de entrada.
            y (Series, optional): Variable objetivo.
            
        Returns:
            self: El selector ajustado.
        """
        if self.verbose:
            print(f"Realizando análisis espectral usando método '{self.method}'...")
        
        # Filtrar solo columnas numéricas
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        
        # Inicializar diccionarios para almacenar resultados
        spectral_power = {}
        frequencies = {}
        acf_values = {}
        
        # Calcular características espectrales para cada columna
        for col in tqdm(numeric_cols, disable=not self.verbose):
            # Eliminar NaNs
            series = X[col].dropna()
            
            if len(series) < 2:
                if self.verbose:
                    print(f"Omitiendo {col}: datos insuficientes después de eliminar NaNs")
                continue
            
            if self.method == 'periodogram':
                # Calcular periodograma
                f, Pxx = periodogram(series, fs=self.fs)
                
                # Guardar resultados
                frequencies[col] = f
                spectral_power[col] = Pxx
                
                # Calcular importancia como la potencia máxima en el espectro
                # (excluyendo la componente DC en f=0)
                if len(Pxx) > 1:
                    spectral_power[col] = np.max(Pxx[1:])
                else:
                    spectral_power[col] = 0
                    
            elif self.method == 'acf':
                # Calcular autocorrelación
                acf_result = acf(series, nlags=min(40, len(series)-1), fft=True)
                
                # Guardar resultados
                acf_values[col] = acf_result
                
                # Calcular importancia como la suma de los valores absolutos de autocorrelación
                # (excluyendo el lag 0 que siempre es 1)
                if len(acf_result) > 1:
                    spectral_power[col] = np.sum(np.abs(acf_result[1:]))
                else:
                    spectral_power[col] = 0
            else:
                raise ValueError(f"Método desconocido: {self.method}. Opciones válidas: 'periodogram', 'acf'")
        
        # Guardar resultados
        self.spectral_power_ = pd.Series(spectral_power)
        self.frequencies_ = frequencies if self.method == 'periodogram' else None
        self.acf_values_ = acf_values if self.method == 'acf' else None
        
        # Establecer importancias
        self.feature_importances_ = self.spectral_power_
        
        # Seleccionar características
        self.selected_features_ = self._select_features()
        
        if self.verbose:
            print(f"Seleccionadas {len(self.selected_features_)} características basadas en análisis espectral")
        
        return self
    
    def plot_spectrum(self, features=None, top_n=5, figsize=(12, 8), save_path=None):
        """
        Visualiza el espectro de potencia para las características seleccionadas.
        
        Args:
            features (list, optional): Lista de características a visualizar.
            top_n (int, optional): Número de características principales a mostrar si features es None.
            figsize (tuple, optional): Tamaño de la figura.
            save_path (str, optional): Ruta para guardar la figura.
            
        Returns:
            matplotlib.figure.Figure: La figura generada.
        """
        if self.method != 'periodogram':
            raise ValueError("Esta visualización solo está disponible para el método 'periodogram'")
        
        if self.frequencies_ is None or self.spectral_power_ is None:
            raise ValueError("El selector debe ser ajustado antes de visualizar")
        
        # Determinar características a visualizar
        if features is None:
            # Usar top_n características con mayor potencia espectral
            features = self.spectral_power_.sort_values(ascending=False).iloc[:top_n].index.tolist()
        
        # Crear figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Graficar espectro para cada característica
        for feature in features:
            if feature in self.frequencies_:
                f = self.frequencies_[feature]
                Pxx = periodogram(pd.Series(self.frequencies_[feature]), fs=self.fs)[1]
                ax.semilogy(f, Pxx, label=feature)
        
        # Añadir títulos y etiquetas
        ax.set_title('Espectro de Potencia')
        ax.set_xlabel('Frecuencia')
        ax.set_ylabel('Densidad Espectral de Potencia')
        ax.legend()
        
        # Ajustar diseño
        plt.tight_layout()
        
        # Guardar figura si se especifica ruta
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def plot_acf(self, features=None, top_n=5, figsize=(12, 8), save_path=None):
        """
        Visualiza la función de autocorrelación para las características seleccionadas.
        
        Args:
            features (list, optional): Lista de características a visualizar.
            top_n (int, optional): Número de características principales a mostrar si features es None.
            figsize (tuple, optional): Tamaño de la figura.
            save_path (str, optional): Ruta para guardar la figura.
            
        Returns:
            matplotlib.figure.Figure: La figura generada.
        """
        if self.method != 'acf':
            raise ValueError("Esta visualización solo está disponible para el método 'acf'")
        
        if self.acf_values_ is None or self.spectral_power_ is None:
            raise ValueError("El selector debe ser ajustado antes de visualizar")
        
        # Determinar características a visualizar
        if features is None:
            # Usar top_n características con mayor potencia espectral
            features = self.spectral_power_.sort_values(ascending=False).iloc[:top_n].index.tolist()
        
        # Crear figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Graficar ACF para cada característica
        for feature in features:
            if feature in self.acf_values_:
                acf_vals = self.acf_values_[feature]
                lags = np.arange(len(acf_vals))
                ax.stem(lags, acf_vals, label=feature, use_line_collection=True)
        
        # Añadir títulos y etiquetas
        ax.set_title('Función de Autocorrelación')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelación')
        ax.legend()
        
        # Ajustar diseño
        plt.tight_layout()
        
        # Guardar figura si se especifica ruta
        if save_path:
            plt.savefig(save_path)
        
        return fig


def create_time_series_selector(method='granger', n_features=None, threshold=None, **kwargs):
    """
    Crea un selector de características específico para series temporales según el método especificado.
    
    Args:
        method (str): Método de selección ('granger', 'pca', 'spectral').
        n_features (int, optional): Número de características a seleccionar.
        threshold (float, optional): Umbral para la selección de características.
        **kwargs: Argumentos adicionales específicos para cada método.
        
    Returns:
        BaseFeatureSelector: El selector de características correspondiente.
    """
    verbose = kwargs.get('verbose', False)
    
    if method == 'granger':
        return GrangerCausalitySelector(
            n_features=n_features,
            threshold=threshold,
            max_lag=kwargs.get('max_lag', 5),
            test=kwargs.get('test', 'ssr_chi2test'),
            verbose=verbose
        )
    elif method == 'pca':
        return PCASelector(
            n_components=n_features,
            variance_threshold=kwargs.get('variance_threshold', 0.95),
            verbose=verbose
        )
    elif method == 'spectral':
        return SpectralSelector(
            n_features=n_features,
            threshold=threshold,
            fs=kwargs.get('fs', 1.0),
            method=kwargs.get('spectral_method', 'periodogram'),
            verbose=verbose
        )
    else:
        raise ValueError(f"Método desconocido: {method}. Opciones válidas: 'granger', 'pca', 'spectral'")
