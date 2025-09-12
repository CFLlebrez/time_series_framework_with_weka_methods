#!/usr/bin/env python3
"""
Métodos de selección de atributos inspirados en Weka para series temporales.

Este módulo implementa métodos de selección de atributos inspirados en Weka,
incluyendo:
- CFS (Correlation-based Feature Selection)
- InfoGain (Information Gain)
- ReliefF
- BestFirst (estrategia de búsqueda)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, pointbiserialr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import math
from tqdm import tqdm
import warnings

# Importar la clase base desde el módulo de correlación
from .correlation_based import BaseFeatureSelector


class PriorityQueue:
    """
    Implementación simple de una cola de prioridad para BestFirst.
    """
    def __init__(self):
        self.queue = []
        
    def isEmpty(self):
        return len(self.queue) == 0
    
    def insert(self, item, priority):
        self.queue.append((item, priority))
        
    def pop(self):
        if self.isEmpty():
            return None
        
        # Encontrar el elemento con mayor prioridad
        max_idx = 0
        max_priority = self.queue[0][1]
        
        for i in range(1, len(self.queue)):
            if self.queue[i][1] > max_priority:
                max_priority = self.queue[i][1]
                max_idx = i
        
        # Eliminar y devolver el elemento con mayor prioridad
        item = self.queue[max_idx]
        del self.queue[max_idx]
        
        return item


class CFSSelector(BaseFeatureSelector):
    """Selector de características basado en CFS (Correlation-based Feature Selection)."""
    
    def __init__(self, n_features=None, threshold=None, max_backtrack=5, verbose=False):
        """
        Inicializa el selector basado en CFS.
        
        Args:
            n_features (int, optional): Número máximo de características a seleccionar.
            threshold (float, optional): Umbral para la selección de características.
            max_backtrack (int, optional): Número máximo de retrocesos en la búsqueda.
            verbose (bool, optional): Si es True, muestra información detallada.
        """
        super().__init__(n_features, threshold, verbose)
        self.max_backtrack = max_backtrack
        self.feature_correlations_ = None
        self.class_correlations_ = None
        
    def fit(self, X, y=None):
        """
        Ajusta el selector usando CFS.
        
        Args:
            X (DataFrame): Datos de entrada.
            y (Series): Variable objetivo.
            
        Returns:
            self: El selector ajustado.
        """
        if y is None:
            raise ValueError("La variable objetivo (y) es requerida para CFS")
        
        # Filtrar solo columnas numéricas
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_numeric = X[numeric_cols]
        
        if self.verbose:
            print("Iniciando selección de características con CFS...")
        
        # Calcular correlaciones característica-clase
        self.class_correlations_ = {}
        
        # Determinar el tipo de correlación a usar según el tipo de variable objetivo
        if np.issubdtype(y.dtype, np.number) and len(np.unique(y)) > 2:
            # Variable objetivo continua - usar correlación de Pearson
            for feature in numeric_cols:
                corr, _ = pearsonr(X_numeric[feature], y)
                self.class_correlations_[feature] = abs(corr)
        else:
            # Variable objetivo categórica o binaria - usar correlación punto-biserial
            # Convertir a binario si es categórico
            if not np.issubdtype(y.dtype, np.number):
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
            else:
                y_encoded = y.values
                
            for feature in numeric_cols:
                try:
                    corr = pointbiserialr(y_encoded, X_numeric[feature].values)
                    self.class_correlations_[feature] = abs(corr.correlation)
                except:
                    # En caso de error (por ejemplo, si todas las características son iguales)
                    self.class_correlations_[feature] = 0.0
        
        # Calcular matriz de correlaciones característica-característica
        corr_matrix = X_numeric.corr().abs()
        self.feature_correlations_ = corr_matrix
        
        # Iniciar búsqueda BestFirst
        selected_features = self._best_first_search(numeric_cols, X_numeric, y)
        
        # Guardar características seleccionadas
        self.selected_features_ = selected_features
        
        # Calcular importancias basadas en correlaciones con la clase
        self.feature_importances_ = pd.Series(self.class_correlations_)
        
        if self.verbose:
            print(f"CFS completado. Seleccionadas {len(self.selected_features_)} características.")
        
        return self
    
    def _calculate_merit(self, subset, X=None, y=None):
        """
        Calcula el mérito de un subconjunto de características.
        
        Args:
            subset (list): Lista de características en el subconjunto.
            X (DataFrame, optional): Datos de entrada (no usado, solo para compatibilidad).
            y (Series, optional): Variable objetivo (no usado, solo para compatibilidad).
            
        Returns:
            float: Valor de mérito del subconjunto.
        """
        if not subset:
            return 0.0
        
        k = len(subset)
        
        # Calcular correlación media característica-clase
        rcf = np.mean([self.class_correlations_[feature] for feature in subset])
        
        # Calcular correlación media característica-característica
        if k > 1:
            rff_sum = 0.0
            count = 0
            
            for i in range(k):
                for j in range(i+1, k):
                    rff_sum += self.feature_correlations_.loc[subset[i], subset[j]]
                    count += 1
            
            rff = rff_sum / count if count > 0 else 0.0
        else:
            rff = 0.0
        
        # Calcular mérito
        denominator = np.sqrt(k + k * (k - 1) * rff)
        
        if denominator == 0:
            return 0.0
        
        return (k * rcf) / denominator
    
    def _best_first_search(self, features, X, y):
        """
        Implementa la búsqueda BestFirst para encontrar el mejor subconjunto de características.
        
        Args:
            features (list): Lista de todas las características disponibles.
            X (DataFrame): Datos de entrada.
            y (Series): Variable objetivo.
            
        Returns:
            list: Lista de características seleccionadas.
        """
        # Inicializar cola de prioridad
        queue = PriorityQueue()
        
        # Inicializar conjunto vacío
        current_subset = []
        current_merit = self._calculate_merit(current_subset)
        
        # Inicializar mejor subconjunto
        best_subset = current_subset.copy()
        best_merit = current_merit
        
        # Inicializar conjuntos visitados
        visited = set()
        visited.add(tuple(sorted(current_subset)))
        
        # Inicializar contador de retrocesos
        backtrack_count = 0
        
        # Inicializar progreso
        if self.verbose:
            pbar = tqdm(total=100, desc="Búsqueda BestFirst")
            progress = 0
        
        while backtrack_count < self.max_backtrack:
            # Generar expansiones (añadir una característica)
            expanded = False
            
            for feature in features:
                if feature not in current_subset:
                    # Crear nuevo subconjunto añadiendo esta característica
                    new_subset = current_subset + [feature]
                    new_subset_tuple = tuple(sorted(new_subset))
                    
                    # Verificar si ya ha sido visitado
                    if new_subset_tuple in visited:
                        continue
                    
                    # Calcular mérito del nuevo subconjunto
                    new_merit = self._calculate_merit(new_subset)
                    
                    # Añadir a la cola de prioridad
                    queue.insert(new_subset, new_merit)
                    
                    # Marcar como visitado
                    visited.add(new_subset_tuple)
                    
                    expanded = True
            
            # Si no se pudo expandir, terminar
            if not expanded and queue.isEmpty():
                break
            
            # Obtener siguiente subconjunto a explorar
            if not queue.isEmpty():
                next_item = queue.pop()
                next_subset, next_merit = next_item
                
                # Actualizar mejor subconjunto si es necesario
                if next_merit > best_merit:
                    best_subset = next_subset
                    best_merit = next_merit
                    backtrack_count = 0  # Reiniciar contador de retrocesos
                else:
                    backtrack_count += 1
                
                current_subset = next_subset
                current_merit = next_merit
            else:
                break
            
            # Actualizar progreso
            if self.verbose:
                new_progress = min(100, int((backtrack_count / self.max_backtrack) * 100))
                if new_progress > progress:
                    pbar.update(new_progress - progress)
                    progress = new_progress
        
        # Cerrar barra de progreso
        if self.verbose:
            pbar.close()
        
        # Limitar número de características si es necesario
        if self.n_features is not None and len(best_subset) > self.n_features:
            # Ordenar por correlación con la clase
            sorted_features = sorted(best_subset, 
                                    key=lambda x: self.class_correlations_[x], 
                                    reverse=True)
            best_subset = sorted_features[:self.n_features]
        
        return best_subset


class InfoGainSelector(BaseFeatureSelector):
    """Selector de características basado en InfoGain (Information Gain)."""
    
    def __init__(self, n_features=None, threshold=None, discretize=True, n_bins=10, verbose=False):
        """
        Inicializa el selector basado en InfoGain.
        
        Args:
            n_features (int, optional): Número de características a seleccionar.
            threshold (float, optional): Umbral para la selección de características.
            discretize (bool, optional): Si es True, discretiza variables continuas.
            n_bins (int, optional): Número de bins para discretización.
            verbose (bool, optional): Si es True, muestra información detallada.
        """
        super().__init__(n_features, threshold, verbose)
        self.discretize = discretize
        self.n_bins = n_bins
    
    def fit(self, X, y=None):
        """
        Ajusta el selector usando InfoGain.
        
        Args:
            X (DataFrame): Datos de entrada.
            y (Series): Variable objetivo.
            
        Returns:
            self: El selector ajustado.
        """
        if y is None:
            raise ValueError("La variable objetivo (y) es requerida para InfoGain")
        
        # Filtrar solo columnas numéricas
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_numeric = X[numeric_cols]
        
        if self.verbose:
            print("Iniciando selección de características con InfoGain...")
        
        # Discretizar la variable objetivo si es continua
        if np.issubdtype(y.dtype, np.number) and len(np.unique(y)) > self.n_bins:
            y_disc = pd.qcut(y, self.n_bins, labels=False, duplicates='drop')
        else:
            # Convertir a categórico si no lo es
            if not np.issubdtype(y.dtype, np.number):
                le = LabelEncoder()
                y_disc = le.fit_transform(y)
            else:
                y_disc = y.values
        
        # Calcular entropía de la clase
        class_entropy = self._calculate_entropy(y_disc)
        
        # Calcular ganancia de información para cada característica
        info_gains = {}
        
        for feature in numeric_cols:
            # Discretizar característica si es necesario
            if self.discretize and len(np.unique(X_numeric[feature])) > self.n_bins:
                try:
                    feature_disc = pd.qcut(X_numeric[feature], self.n_bins, labels=False, duplicates='drop')
                except:
                    # Si falla qcut (por ejemplo, con muchos valores duplicados), usar cut
                    feature_disc = pd.cut(X_numeric[feature], self.n_bins, labels=False)
            else:
                feature_disc = X_numeric[feature].values
            
            # Calcular entropía condicional
            conditional_entropy = self._calculate_conditional_entropy(feature_disc, y_disc)
            
            # Calcular ganancia de información
            info_gain = class_entropy - conditional_entropy
            info_gains[feature] = info_gain
        
        # Convertir a Series
        self.feature_importances_ = pd.Series(info_gains)
        
        # Seleccionar características
        if self.threshold is not None:
            # Seleccionar características con ganancia de información por encima del umbral
            self.selected_features_ = [feature for feature, gain in info_gains.items() 
                                     if gain >= self.threshold]
        elif self.n_features is not None:
            # Seleccionar las n_features con mayor ganancia de información
            sorted_features = sorted(info_gains.items(), key=lambda x: x[1], reverse=True)
            self.selected_features_ = [feature for feature, _ in sorted_features[:self.n_features]]
        else:
            # Por defecto, seleccionar características con ganancia positiva
            self.selected_features_ = [feature for feature, gain in info_gains.items() 
                                     if gain > 0]
        
        if self.verbose:
            print(f"InfoGain completado. Seleccionadas {len(self.selected_features_)} características.")
        
        return self
    
    def _calculate_entropy(self, y):
        """
        Calcula la entropía de una variable.
        
        Args:
            y (array): Variable para calcular la entropía.
            
        Returns:
            float: Valor de entropía.
        """
        # Calcular probabilidades
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        
        # Calcular entropía
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        return entropy
    
    def _calculate_conditional_entropy(self, x, y):
        """
        Calcula la entropía condicional H(Y|X).
        
        Args:
            x (array): Variable condicionante.
            y (array): Variable objetivo.
            
        Returns:
            float: Valor de entropía condicional.
        """
        # Valores únicos de x
        x_values = np.unique(x)
        
        # Calcular entropía condicional
        conditional_entropy = 0
        
        for value in x_values:
            # Índices donde x == value
            indices = np.where(x == value)[0]
            
            # Probabilidad de x == value
            p_x = len(indices) / len(x)
            
            # Valores de y donde x == value
            y_given_x = y[indices]
            
            # Entropía de y dado x == value
            entropy_y_given_x = self._calculate_entropy(y_given_x)
            
            # Añadir a la entropía condicional
            conditional_entropy += p_x * entropy_y_given_x
        
        return conditional_entropy


class ReliefFSelector(BaseFeatureSelector):
    """Selector de características basado en ReliefF."""
    
    def __init__(self, n_features=None, threshold=None, n_neighbors=10, 
                 sample_size=None, discrete_threshold=10, n_jobs=1, verbose=False):
        """
        Inicializa el selector basado en ReliefF.
        
        Args:
            n_features (int, optional): Número de características a seleccionar.
            threshold (float, optional): Umbral para la selección de características.
            n_neighbors (int, optional): Número de vecinos a considerar.
            sample_size (int, optional): Tamaño de la muestra a usar (None = usar todos).
            discrete_threshold (int): Umbral para considerar una característica como discreta.
            n_jobs (int): Número de trabajos paralelos para k-NN.
            verbose (bool, optional): Si es True, muestra información detallada.
        """
        super().__init__(n_features, threshold, verbose)
        self.n_neighbors = n_neighbors
        self.sample_size = sample_size
        self.discrete_threshold = discrete_threshold
        self.n_jobs = n_jobs
    
    def fit(self, X, y=None):
        """
        Ajusta el selector usando ReliefF.
        
        Args:
            X (DataFrame): Datos de entrada.
            y (Series): Variable objetivo.
            
        Returns:
            self: El selector ajustado.
        """
        if y is None:
            raise ValueError("La variable objetivo (y) es requerida para ReliefF")
        
        # Filtrar solo columnas numéricas
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_numeric = X[numeric_cols]
        
        if self.verbose:
            print("Iniciando selección de características con ReliefF...")
        
        # Determinar si la variable objetivo es discreta o continua
        if len(np.unique(y)) <= self.discrete_threshold or not np.issubdtype(y.dtype, np.number):
            # Discreto - clasificación
            self._fit_classification(X_numeric, y)
        else:
            # Continuo - regresión
            self._fit_regression(X_numeric, y)
        
        # Seleccionar características
        if self.threshold is not None:
            # Seleccionar características con puntuación por encima del umbral
            self.selected_features_ = [feature for feature, score in self.feature_importances_.items() 
                                     if score >= self.threshold]
        elif self.n_features is not None:
            # Seleccionar las n_features con mayor puntuación
            sorted_features = sorted(self.feature_importances_.items(), key=lambda x: x[1], reverse=True)
            self.selected_features_ = [feature for feature, _ in sorted_features[:self.n_features]]
        else:
            # Por defecto, seleccionar características con puntuación positiva
            self.selected_features_ = [feature for feature, score in self.feature_importances_.items() 
                                     if score > 0]
        
        if self.verbose:
            print(f"ReliefF completado. Seleccionadas {len(self.selected_features_)} características.")
        
        return self
    
    def _fit_classification(self, X, y):
        """
        Ajusta ReliefF para problemas de clasificación.
        
        Args:
            X (DataFrame): Datos de entrada.
            y (Series): Variable objetivo (clases).
        """
        # Convertir a numpy arrays
        X_values = X.values
        feature_names = X.columns.tolist()
        
        # Codificar clases si es necesario
        if not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y_values = le.fit_transform(y)
        else:
            y_values = y.values
        
        # Determinar tamaño de muestra
        n_samples = len(X)
        if self.sample_size is not None and self.sample_size < n_samples:
            sample_size = self.sample_size
            # Muestrear índices estratificados
            indices = []
            for c in np.unique(y_values):
                class_indices = np.where(y_values == c)[0]
                class_sample_size = int(sample_size * len(class_indices) / n_samples)
                indices.extend(np.random.choice(class_indices, class_sample_size, replace=False))
            
            # Asegurar que tenemos exactamente sample_size
            if len(indices) < sample_size:
                remaining = np.setdiff1d(np.arange(n_samples), indices)
                indices.extend(np.random.choice(remaining, sample_size - len(indices), replace=False))
            elif len(indices) > sample_size:
                indices = np.random.choice(indices, sample_size, replace=False)
            
            X_sample = X_values[indices]
            y_sample = y_values[indices]
        else:
            X_sample = X_values
            y_sample = y_values
            indices = np.arange(n_samples)
        
        # Inicializar pesos de características
        weights = np.zeros(X.shape[1])
        
        # Encontrar vecinos más cercanos para cada instancia
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nn = NearestNeighbors(n_neighbors=self.n_neighbors+1, n_jobs=self.n_jobs)
            nn.fit(X_values)
            distances, neighbors = nn.kneighbors(X_sample)
        
        # Eliminar la primera columna (la instancia misma)
        neighbors = neighbors[:, 1:]
        distances = distances[:, 1:]
        
        # Calcular pesos
        n_features = X.shape[1]
        n_instances = len(X_sample)
        
        # Determinar características discretas
        is_discrete = np.array([len(np.unique(X_values[:, i])) <= self.discrete_threshold 
                              for i in range(n_features)])
        
        # Calcular pesos para cada instancia
        for i in range(n_instances):
            instance = X_sample[i]
            instance_class = y_sample[i]
            
            # Encontrar vecinos de la misma clase (hits) y de diferente clase (misses)
            hit_indices = [j for j, idx in enumerate(neighbors[i]) 
                         if y_values[idx] == instance_class]
            miss_indices = [j for j, idx in enumerate(neighbors[i]) 
                          if y_values[idx] != instance_class]
            
            # Calcular pesos para cada característica
            for f in range(n_features):
                # Diferencia con hits
                hit_diff = 0
                for j in hit_indices:
                    neighbor_idx = neighbors[i, j]
                    if is_discrete[f]:
                        hit_diff += int(instance[f] != X_values[neighbor_idx, f])
                    else:
                        hit_diff += abs(instance[f] - X_values[neighbor_idx, f])
                
                hit_diff /= len(hit_indices) if hit_indices else 1
                
                # Diferencia con misses
                miss_diff = 0
                for j in miss_indices:
                    neighbor_idx = neighbors[i, j]
                    if is_discrete[f]:
                        miss_diff += int(instance[f] != X_values[neighbor_idx, f])
                    else:
                        miss_diff += abs(instance[f] - X_values[neighbor_idx, f])
                
                miss_diff /= len(miss_indices) if miss_indices else 1
                
                # Actualizar peso
                weights[f] += miss_diff - hit_diff
        
        # Normalizar pesos
        weights /= n_instances
        
        # Guardar importancias
        self.feature_importances_ = pd.Series(weights, index=feature_names)
    
    def _fit_regression(self, X, y):
        """
        Ajusta ReliefF para problemas de regresión (RReliefF).
        
        Args:
            X (DataFrame): Datos de entrada.
            y (Series): Variable objetivo (valores continuos).
        """
        # Convertir a numpy arrays
        X_values = X.values
        feature_names = X.columns.tolist()
        y_values = y.values
        
        # Determinar tamaño de muestra
        n_samples = len(X)
        if self.sample_size is not None and self.sample_size < n_samples:
            indices = np.random.choice(n_samples, self.sample_size, replace=False)
            X_sample = X_values[indices]
            y_sample = y_values[indices]
        else:
            X_sample = X_values
            y_sample = y_values
            indices = np.arange(n_samples)
        
        # Inicializar pesos de características
        weights = np.zeros(X.shape[1])
        
        # Encontrar vecinos más cercanos para cada instancia
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nn = NearestNeighbors(n_neighbors=self.n_neighbors+1, n_jobs=self.n_jobs)
            nn.fit(X_values)
            distances, neighbors = nn.kneighbors(X_sample)
        
        # Eliminar la primera columna (la instancia misma)
        neighbors = neighbors[:, 1:]
        distances = distances[:, 1:]
        
        # Calcular pesos
        n_features = X.shape[1]
        n_instances = len(X_sample)
        
        # Determinar características discretas
        is_discrete = np.array([len(np.unique(X_values[:, i])) <= self.discrete_threshold 
                              for i in range(n_features)])
        
        # Inicializar acumuladores para RReliefF
        diff_weighted_sum = np.zeros(n_features)
        weight_sum = 0
        
        # Calcular pesos para cada instancia
        for i in range(n_instances):
            instance = X_sample[i]
            instance_value = y_sample[i]
            
            for j in range(self.n_neighbors):
                neighbor_idx = neighbors[i, j]
                neighbor = X_values[neighbor_idx]
                neighbor_value = y_values[neighbor_idx]
                
                # Calcular distancia en el espacio de salida (normalizada)
                y_diff = abs(instance_value - neighbor_value)
                
                # Calcular peso de la instancia (decae con la distancia)
                weight = np.exp(-y_diff)
                weight_sum += weight
                
                # Calcular diferencias para cada característica
                for f in range(n_features):
                    if is_discrete[f]:
                        diff = int(instance[f] != neighbor[f])
                    else:
                        diff = abs(instance[f] - neighbor[f])
                    
                    # Acumular diferencia ponderada
                    diff_weighted_sum[f] += diff * weight
        
        # Calcular pesos finales
        if weight_sum > 0:
            weights = diff_weighted_sum / weight_sum
        
        # Normalizar pesos al rango [0, 1]
        if np.max(weights) > 0:
            weights = weights / np.max(weights)
        
        # Guardar importancias
        self.feature_importances_ = pd.Series(weights, index=feature_names)


def create_weka_inspired_selector(method='cfs', n_features=None, threshold=None, **kwargs):
    """
    Crea un selector de características inspirado en Weka.
    
    Args:
        method (str): Método de selección ('cfs', 'infogain', 'relieff').
        n_features (int, optional): Número de características a seleccionar.
        threshold (float, optional): Umbral para la selección de características.
        **kwargs: Argumentos adicionales específicos para cada método.
        
    Returns:
        BaseFeatureSelector: El selector de características correspondiente.
    """
    verbose = kwargs.get('verbose', False)
    
    if method == 'cfs':
        return CFSSelector(
            n_features=n_features,
            threshold=threshold,
            max_backtrack=kwargs.get('max_backtrack', 5),
            verbose=verbose
        )
    elif method == 'infogain':
        return InfoGainSelector(
            n_features=n_features,
            threshold=threshold,
            discretize=kwargs.get('discretize', True),
            n_bins=kwargs.get('n_bins', 10),
            verbose=verbose
        )
    elif method == 'relieff':
        return ReliefFSelector(
            n_features=n_features,
            threshold=threshold,
            n_neighbors=kwargs.get('n_neighbors', 10),
            sample_size=kwargs.get('sample_size', None),
            discrete_threshold=kwargs.get('discrete_threshold', 10),
            n_jobs=kwargs.get('n_jobs', 1),
            verbose=verbose
        )
    else:
        raise ValueError(f"Método desconocido: {method}. Opciones válidas: 'cfs', 'infogain', 'relieff'")
