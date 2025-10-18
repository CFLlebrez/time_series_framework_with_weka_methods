## \_\_init__.py
- create_feature_selector():<br>
-**correlation**: pearson, ccf, mutual_info<br>
-**model**: random_forest, lasso, elastic_net, rfe<br>
-**time series**: granger, pca, spectral<br>
-**automatic**: sequential, genetic<br>
-**sklearn**: read sklearn_filter.py section<br>
-**weka**: read weka_inspired.py section<br>

- prepare_data_for_selection():<br>
-Removes steps and target columns (date excluded before)

- generate_feature_importance_report():<br>
-Generates csv with importances and bool indicating if selected<br>
-Generates bar graph with features and importances sorted

- filter_data_with_selected_features():<br>
-Generates csv just with selected features (not used)

- select_features() MAIN FUNCTION:<br>
-separates the time_col **here is where step cols should be saved if wanted**.<br>
-calls prepare_data_for_selection()<br>
-calls create_feature_selector()<br>
-fits selector to prepared data and target<br>
-generates report<br>
-generates filtered csv appending the date again.

## correlation_based.py
- ### class BaseFeatureSelector:
n_features, threshold, feature_importances_, selected_features_<br><br>
-\_\_init__()<br>
-fit() (each specific selector implements this)<br>
-transform & fit_transform not used<br>
-get_selected_features(), _select_features() get, initialize selected_features_<br>
**-plot_feature_importances() implemented in \_\_init__.py**

- ### class PearsonCorrelationSelector:
(specific attributes) absolute <br><br>
-\_\_init__()<br>
-fit(): y to Series, pearsonr -> correlations(importances) to Series<br>

- ### class CrossCorrelationSelector (CCF):
absolute, max_lag (and best_lags) <br><br>
-\_\_init__()<br>
-fit(): y to Series, correlations(importances) to Series<br>
**-As max_lag was forced to 0, CCF is equivalent to Pearson.**<br>
-get_best_lags(): normal get<br>
-plot_ccf_heatmap() (not used)<br>

- ### class MutualInformationSelector:
n_neighbors <br><br>
-\_\_init__()<br>
-fit(): numeric cols, y, n_neighbors -> mi_value to Series<br>

**create_correlation_selector**<br>

## automatic_selection.py
- ### class SequentialFeatureSelector:
direction, scoring, cv, estimator <br><br>
-\_\_init__()<br>
-fit(): forward(none and adds)/backward(all and removes) candidate sets -> best score (using evaluate) <br>
-\_evaluate_feature_set(): cross_val_score(cv, scoring)<br>
-get and plot _selection_history() (plot for external use, not used in this file).<br>

- ### class GeneticFeatureSelector:
population_size, generations, crossover_prob, mutation_prob, tournament_size, scoring, cv, estimator, random_state(seed)<br><br>
-\_\_init__()<br>
-fit(): aplicar genéticos.<br>
-_initialize_population.<br>
-_evaluate_individual.<br>
-_selection.<br>
-_crossover.<br>
-_mutation.<br>
-plot_evolution.<br>

**create_automatic_selector**<br>

## model_based.py
- ### class RandomForestSelector:
n_estimators, max_depth, random_state <br><br>
-\_\_init__()<br>
-fit(): RandomForestRegressor -> feature_importances as Series, select_features().<br>
-plot_feature_importances(): plot.<br>

- ### class LassoSelector:
alpha (regularization), max_iter, random_state <br><br>
-\_\_init__()<br>
-fit(): scales data, Lasso -> feature_importances (abs(coefs)) as Series, select_features().<br>
-get_coefficients(): extracts coefficients from the model.<br>

- ### class ElasticNetSelector:
alpha (regularization), l1_ratio (L1% in L1 and L2 mix. Lasso is 1 and Ridge is 0), max_iter, random_state <br><br>
-\_\_init__()<br>
-fit(): scales data, ElasticNet -> feature importances (abs(coefs)) as Series, select_features()<br>
-get_coefficients(): extracts coefficients from the model.<br>

- ### class RFESelector:
step, estimator (or RandomForest) <br><br>
-\_\_init__()<br>
-fit(): scales data, RFE -> feature importances (inverts for greater~more important) as Series, select_features()<br>
-get_coefficients(): extracts coefficients from the model.<br>

**create_model_selector**<br>
