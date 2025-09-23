#!/usr/bin/env python3
"""
Métodos de selección automática de atributos para series temporales.

Este módulo implementa métodos de selección automática de atributos para series temporales,
incluyendo:
- Selección secuencial hacia adelante/atrás (SFS/SBS)
- Algoritmos genéticos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import random
import warnings

# Importar la clase base desde el módulo de correlación
from .correlation_based import BaseFeatureSelector


class SequentialFeatureSelector(BaseFeatureSelector):
    """Selector de características basado en selección secuencial."""
    
    def __init__(self, n_features=None, direction='forward', scoring='neg_mean_squared_error', 
                 cv=5, estimator=None, verbose=False):
        """
        Inicializa el selector basado en selección secuencial.
        
        Args:
            n_features (int, optional): Número de características a seleccionar.
            direction (str, optional): Dirección de selección ('forward' o 'backward').
            scoring (str, optional): Métrica de evaluación para selección.
            cv (int, optional): Número de folds para validación cruzada.
            estimator (object, optional): Estimador base para evaluación.
            verbose (bool, optional): Si es True, muestra información detallada.
        """
        super().__init__(n_features, None, verbose)  # No usa threshold
        self.direction = direction
        self.scoring = scoring
        self.cv = cv
        self.estimator = estimator or LinearRegression()
        self.feature_scores_ = None
        self.selection_history_ = None
    
    def fit(self, X, y=None):
        """
        Ajusta el selector usando selección secuencial.
        
        Args:
            X (DataFrame): Datos de entrada.
            y (Series): Variable objetivo.
            
        Returns:
            self: El selector ajustado.
        """
        if y is None:
            raise ValueError("La variable objetivo (y) es requerida para este selector")
        
        if self.n_features is None:
            raise ValueError("El número de características (n_features) es requerido para selección secuencial")
        
        # Filtrar solo columnas numéricas
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_numeric = X[numeric_cols]
        
        # Inicializar variables
        n_total_features = len(numeric_cols)
        
        if self.direction == 'forward':
            if self.verbose:
                print(f"Iniciando selección secuencial hacia adelante para seleccionar {self.n_features} características...")
            
            # Inicializar con conjunto vacío
            selected_features = []
            remaining_features = numeric_cols.copy()
            
        elif self.direction == 'backward':
            if self.verbose:
                print(f"Iniciando selección secuencial hacia atrás para seleccionar {self.n_features} características...")
            
            # Inicializar con todas las características
            selected_features = numeric_cols.copy()
            remaining_features = []
            
        else:
            raise ValueError(f"Dirección desconocida: {self.direction}. Opciones válidas: 'forward', 'backward'")
        
        # Inicializar historial de selección y puntuaciones
        selection_history = []
        feature_scores = {feature: 0.0 for feature in numeric_cols}
        
        # Realizar selección secuencial
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if self.direction == 'forward':
                # Selección hacia adelante
                pbar = tqdm(range(self.n_features), disable=not self.verbose)
                for _ in pbar:
                    if not remaining_features:
                        break
                    
                    best_score = float('-inf')
                    best_feature = None
                    
                    # Evaluar cada característica restante
                    for feature in remaining_features:
                        # Crear conjunto de características candidato
                        candidate_features = selected_features + [feature]
                        
                        # Evaluar modelo con este conjunto
                        score = self._evaluate_feature_set(X_numeric[candidate_features], y)
                        
                        # Actualizar mejor característica
                        if score > best_score:
                            best_score = score
                            best_feature = feature
                    
                    if best_feature is not None:
                        # Añadir mejor característica al conjunto seleccionado
                        selected_features.append(best_feature)
                        remaining_features.remove(best_feature)
                        
                        # Actualizar puntuación de la característica
                        feature_scores[best_feature] = best_score
                        
                        # Actualizar historial
                        selection_history.append((best_feature, best_score))
                        
                        if self.verbose:
                            pbar.set_description(f"Añadida: {best_feature} (score: {best_score:.4f})")
            
            else:  # backward
                # Selección hacia atrás
                pbar = tqdm(range(n_total_features - self.n_features), disable=not self.verbose)
                for _ in pbar:
                    if len(selected_features) <= self.n_features:
                        break
                    
                    best_score = float('-inf')
                    worst_feature = None
                    
                    # Evaluar eliminación de cada característica
                    for feature in selected_features:
                        # Crear conjunto de características candidato
                        candidate_features = [f for f in selected_features if f != feature]
                        
                        # Evaluar modelo con este conjunto
                        score = self._evaluate_feature_set(X_numeric[candidate_features], y)
                        
                        # Actualizar peor característica (la que al eliminarla da mejor puntuación)
                        if score > best_score:
                            best_score = score
                            worst_feature = feature
                    
                    if worst_feature is not None:
                        # Eliminar peor característica del conjunto seleccionado
                        selected_features.remove(worst_feature)
                        remaining_features.append(worst_feature)
                        
                        # Actualizar puntuación de la característica (negativa para indicar que fue eliminada)
                        feature_scores[worst_feature] = -best_score
                        
                        # Actualizar historial
                        selection_history.append((worst_feature, -best_score))
                        
                        if self.verbose:
                            pbar.set_description(f"Eliminada: {worst_feature} (score: {best_score:.4f})")
        
        # Guardar resultados
        self.selected_features_ = selected_features
        self.feature_scores_ = pd.Series(feature_scores)
        self.selection_history_ = selection_history
        
        # Calcular importancias basadas en puntuaciones
        if self.direction == 'forward':
            self.feature_importances_ = self.feature_scores_
        else:  # backward
            # Para selección hacia atrás, las importancias son inversas a las puntuaciones
            self.feature_importances_ = -self.feature_scores_
        
        if self.verbose:
            print(f"Selección secuencial completada. Seleccionadas {len(self.selected_features_)} características.")
        
        return self
    
    def _evaluate_feature_set(self, X_subset, y):
        """
        Evalúa un conjunto de características usando validación cruzada.
        
        Args:
            X_subset (DataFrame): Subconjunto de datos con las características seleccionadas.
            y (Series): Variable objetivo.
            
        Returns:
            float: Puntuación media de validación cruzada.
        """
        # Realizar validación cruzada
        scores = cross_val_score(
            self.estimator, X_subset, y, 
            cv=self.cv, scoring=self.scoring
        )
        
        # Devolver puntuación  (neg para pasar de neg MSE a MSE)
        return -np.mean(scores)
    
    def get_selection_history(self):
        """
        Devuelve el historial de selección de características.
        
        Returns:
            list: Lista de tuplas (característica, puntuación) en orden de selección.
        """
        if self.selection_history_ is None:
            raise ValueError("El selector debe ser ajustado antes de obtener el historial")
        return self.selection_history_
    
    def plot_selection_history(self, figsize=(10, 6), save_path=None):
        """
        Visualiza el historial de selección de características.
        
        Args:
            figsize (tuple, optional): Tamaño de la figura.
            save_path (str, optional): Ruta para guardar la figura.
            
        Returns:
            matplotlib.figure.Figure: La figura generada.
        """
        if self.selection_history_ is None:
            raise ValueError("El selector debe ser ajustado antes de visualizar")
        
        # Extraer características y puntuaciones
        features, scores = zip(*self.selection_history_)
        
        # Crear figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Crear gráfico de línea
        ax.plot(range(1, len(scores) + 1), scores, 'o-')
        
        # Añadir títulos y etiquetas
        if self.direction == 'forward':
            ax.set_title('Historial de Selección Secuencial Hacia Adelante')
        else:
            ax.set_title('Historial de Selección Secuencial Hacia Atrás')
        ax.set_xlabel('Número de Características')
        ax.set_ylabel('Puntuación')
        
        # Añadir etiquetas de características
        for i, (feature, score) in enumerate(self.selection_history_):
            ax.annotate(feature, (i + 1, score), textcoords="offset points", 
                       xytext=(0, 10), ha='center')
        
        # Ajustar diseño
        plt.tight_layout()
        
        # Guardar figura si se especifica ruta
        if save_path:
            plt.savefig(save_path)
        
        return fig


class GeneticFeatureSelector(BaseFeatureSelector):
    """Selector de características basado en algoritmos genéticos."""
    
    def __init__(self, n_features=None, population_size=50, generations=20, 
                 crossover_prob=0.8, mutation_prob=0.1, tournament_size=3,
                 scoring='neg_mean_squared_error', cv=5, estimator=None, 
                 random_state=42, verbose=False):
        """
        Inicializa el selector basado en algoritmos genéticos.
        
        Args:
            n_features (int, optional): Número objetivo de características a seleccionar.
            population_size (int, optional): Tamaño de la población.
            generations (int, optional): Número de generaciones.
            crossover_prob (float, optional): Probabilidad de cruce.
            mutation_prob (float, optional): Probabilidad de mutación.
            tournament_size (int, optional): Tamaño del torneo para selección.
            scoring (str, optional): Métrica de evaluación.
            cv (int, optional): Número de folds para validación cruzada.
            estimator (object, optional): Estimador base para evaluación.
            random_state (int, optional): Semilla para reproducibilidad.
            verbose (bool, optional): Si es True, muestra información detallada.
        """
        super().__init__(n_features, None, verbose)  # No usa threshold
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.scoring = scoring
        self.cv = TimeSeriesSplit(n_splits= cv if (cv.is_integer()) else 5)
        self.estimator = estimator or RandomForestRegressor(n_estimators=50, random_state=random_state)
        self.random_state = random_state
        self.best_individual_ = None
        self.best_score_ = None
        self.evolution_history_ = None
    
    def fit(self, X, y=None):
        """
        Ajusta el selector usando algoritmos genéticos.
        
        Args:
            X (DataFrame): Datos de entrada.
            y (Series): Variable objetivo.
            
        Returns:
            self: El selector ajustado.
        """
        if y is None:
            raise ValueError("La variable objetivo (y) es requerida para este selector")
        
        # Filtrar solo columnas numéricas
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_numeric = X[numeric_cols]
        
        # Establecer semilla aleatoria
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Inicializar variables
        n_features = len(numeric_cols)
        feature_names = numeric_cols
        
        if self.verbose:
            print(f"Iniciando algoritmo genético con población de {self.population_size} y {self.generations} generaciones...")
        
        # Inicializar población
        population = self._initialize_population(n_features)
        
        # Evaluar población inicial
        fitness_scores = [self._evaluate_individual(ind, X_numeric, y, feature_names) for ind in population]
        
        # Inicializar historial de evolución
        evolution_history = []
        best_scores = []
        avg_scores = []
        
        # Evolucionar población
        for generation in tqdm(range(self.generations), disable=not self.verbose):
            # Seleccionar padres
            parents = self._selection(population, fitness_scores)
            
            # Crear nueva población
            new_population = []
            
            # Aplicar operadores genéticos
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    parent1, parent2 = parents[i], parents[i + 1]
                    
                    # Cruce
                    if random.random() < self.crossover_prob:
                        child1, child2 = self._crossover(parent1, parent2)
                    else:
                        child1, child2 = parent1.copy(), parent2.copy()
                    
                    # Mutación
                    child1 = self._mutation(child1)
                    child2 = self._mutation(child2)
                    
                    new_population.extend([child1, child2])
            
            # Asegurar que la población mantiene su tamaño
            if len(new_population) > self.population_size:
                new_population = new_population[:self.population_size]
            
            # Evaluar nueva población
            fitness_scores = [self._evaluate_individual(ind, X_numeric, y, feature_names) for ind in new_population]
            
            # Actualizar población
            population = new_population
            
            # Registrar estadísticas
            best_idx = np.argmax(fitness_scores)
            best_score = fitness_scores[best_idx]
            avg_score = np.mean(fitness_scores)
            
            best_scores.append(best_score)
            avg_scores.append(avg_score)
            
            if self.verbose and (generation % 5 == 0 or generation == self.generations - 1):
                n_selected = sum(population[best_idx])
                print(f"Generación {generation+1}: Mejor puntuación = {best_score:.4f}, "
                      f"Promedio = {avg_score:.4f}, Características seleccionadas = {n_selected}")
        
        # Obtener mejor individuo
        best_idx = np.argmax(fitness_scores)
        self.best_individual_ = population[best_idx]
        self.best_score_ = fitness_scores[best_idx]
        
        # Guardar historial de evolución
        self.evolution_history_ = {
            'best_scores': best_scores,
            'avg_scores': avg_scores
        }
        
        # Seleccionar características basadas en el mejor individuo
        self.selected_features_ = [feature for i, feature in enumerate(feature_names) if self.best_individual_[i] == 1]
        
        # Calcular importancias basadas en frecuencia de selección en la población final
        feature_freq = np.sum(population, axis=0) / len(population)
        self.feature_importances_ = pd.Series(feature_freq, index=feature_names)
        
        if self.verbose:
            print(f"Algoritmo genético completado. Seleccionadas {len(self.selected_features_)} características.")
        
        return self
    
    def _initialize_population(self, n_features):
        """
        Inicializa la población con individuos aleatorios.
        
        Args:
            n_features (int): Número total de características.
            
        Returns:
            list: Lista de individuos (arrays binarios).
        """
        population = []
        
        for _ in range(self.population_size):
            # Crear individuo aleatorio
            if self.n_features is not None:
                # Seleccionar exactamente n_features características
                individual = np.zeros(n_features, dtype=int)
                indices = np.random.choice(n_features, self.n_features, replace=False)
                individual[indices] = 1
            else:
                # Seleccionar un número aleatorio de características
                individual = np.random.randint(0, 2, n_features)
            
            population.append(individual)
        
        return population
    
    def _evaluate_individual(self, individual, X, y, feature_names):
        """
        Evalúa un individuo usando validación cruzada.
        
        Args:
            individual (array): Array binario que representa las características seleccionadas.
            X (DataFrame): Datos de entrada.
            y (Series): Variable objetivo.
            feature_names (list): Nombres de las características.
            
        Returns:
            float: Puntuación de fitness.
        """
        # Si no hay características seleccionadas, devolver puntuación mínima
        if sum(individual) == 0:
            return float('-inf')
        
        # Seleccionar características
        selected_features = [feature for i, feature in enumerate(feature_names) if individual[i] == 1]
        X_subset = X[selected_features]
        
        # Realizar validación cruzada
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(
                self.estimator, X_subset, y, 
                cv=self.cv, scoring=self.scoring
            )
        
        # Calcular puntuación media
        cv_score = np.mean(scores)
        
        # Penalizar si el número de características es muy diferente del objetivo
        if self.n_features is not None:
            n_selected = sum(individual)
            penalty = 1.0 - 0.1 * abs(n_selected - self.n_features) / self.n_features
            penalty = max(0.5, penalty)  # Limitar la penalización
            cv_score = cv_score * penalty
        
        return cv_score
    
    def _selection(self, population, fitness_scores):
        """
        Selecciona individuos para reproducción usando torneo.
        
        Args:
            population (list): Lista de individuos.
            fitness_scores (list): Lista de puntuaciones de fitness.
            
        Returns:
            list: Lista de individuos seleccionados.
        """
        selected = []
        
        for _ in range(len(population)):
            # Seleccionar individuos para torneo
            tournament_idx = np.random.choice(len(population), self.tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_idx]
            
            # Seleccionar el mejor
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        return selected
    
    def _crossover(self, parent1, parent2):
        """
        Realiza cruce entre dos padres.
        
        Args:
            parent1 (array): Primer padre.
            parent2 (array): Segundo padre.
            
        Returns:
            tuple: Dos hijos resultantes del cruce.
        """
        # Punto de cruce aleatorio
        crossover_point = random.randint(1, len(parent1) - 1)
        
        # Crear hijos
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        
        return child1, child2
    
    def _mutation(self, individual):
        """
        Aplica mutación a un individuo.
        
        Args:
            individual (array): Individuo a mutar.
            
        Returns:
            array: Individuo mutado.
        """
        for i in range(len(individual)):
            if random.random() < self.mutation_prob:
                individual[i] = 1 - individual[i]  # Invertir bit
        
        return individual
    
    def plot_evolution(self, figsize=(10, 6), save_path=None):
        """
        Visualiza la evolución del algoritmo genético.
        
        Args:
            figsize (tuple, optional): Tamaño de la figura.
            save_path (str, optional): Ruta para guardar la figura.
            
        Returns:
            matplotlib.figure.Figure: La figura generada.
        """
        if self.evolution_history_ is None:
            raise ValueError("El selector debe ser ajustado antes de visualizar")
        
        # Extraer historiales
        best_scores = self.evolution_history_['best_scores']
        avg_scores = self.evolution_history_['avg_scores']
        generations = range(1, len(best_scores) + 1)
        
        # Crear figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Crear gráficos de línea
        ax.plot(generations, best_scores, 'b-', label='Mejor puntuación')
        ax.plot(generations, avg_scores, 'r-', label='Puntuación promedio')
        
        # Añadir títulos y etiquetas
        ax.set_title('Evolución del Algoritmo Genético')
        ax.set_xlabel('Generación')
        ax.set_ylabel('Puntuación')
        ax.legend()
        
        # Ajustar diseño
        plt.tight_layout()
        
        # Guardar figura si se especifica ruta
        if save_path:
            plt.savefig(save_path)
        
        return fig


def create_automatic_selector(method='sequential', n_features=None, **kwargs):
    """
    Crea un selector automático de características según el método especificado.
    
    Args:
        method (str): Método de selección ('sequential', 'genetic').
        n_features (int, optional): Número de características a seleccionar.
        **kwargs: Argumentos adicionales específicos para cada método.
        
    Returns:
        BaseFeatureSelector: El selector de características correspondiente.
    """
    verbose = kwargs.get('verbose', False)
    
    if method == 'sequential':
        return SequentialFeatureSelector(
            n_features=n_features,
            direction=kwargs.get('direction', 'forward'),
            scoring=kwargs.get('scoring', 'neg_mean_squared_error'),
            cv=kwargs.get('cv', 5),
            estimator=kwargs.get('estimator', None),
            verbose=verbose
        )
    elif method == 'genetic':
        return GeneticFeatureSelector(
            n_features=n_features,
            population_size=kwargs.get('population_size', 50),
            generations=kwargs.get('generations', 20),
            crossover_prob=kwargs.get('crossover_prob', 0.8),
            mutation_prob=kwargs.get('mutation_prob', 0.1),
            tournament_size=kwargs.get('tournament_size', 3),
            scoring=kwargs.get('scoring', 'neg_mean_squared_error'),
            cv=kwargs.get('cv', 5),
            estimator=kwargs.get('estimator', None),
            random_state=kwargs.get('random_state', 42),
            verbose=verbose
        )
    else:
        raise ValueError(f"Método desconocido: {method}. Opciones válidas: 'sequential', 'genetic'")
