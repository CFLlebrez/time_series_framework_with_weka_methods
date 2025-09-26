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
- get_best_lags(): normal get
- plot_ccf_heatmap() (not used)

- ### class MutualInformationSelector:
n_neighbors <br><br>
-\_\_init__()<br>
-fit(): numeric cols, y, n_neighbors -> mi_value to Series<br>

**create_correlation_selector**
## automatic_selection.py
