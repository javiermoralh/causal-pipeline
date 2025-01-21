import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone
from sklearn.preprocessing import label_binarize
from typing import List, Dict, Union, Optional
from sklearn.model_selection import BaseCrossValidator
from config import TREATMENT
from modules.dimensionality_reduction.funcs import compute_cv_score


class SFSCV(BaseEstimator, TransformerMixin):

    """

    Sequential Forward Selection with Cross-Validation (SFSCV).

    

    This class implements a feature selection algorithm that works with any type of supervised 
    learning task (binary classification, multiclass classification, or regression). It 
    iteratively adds features based on their impact on model performance, continuing
    until no more features can be added that improve the performance.
    
    The algorithm automatically detects the type of prediction needed based on the scorer
    and model capabilities, making it versatile across different types of problems.

    Attributes:
        selected_features_: List[str]
            Features selected during fitting
        feature_scores_: Dict[str, float]
            Mapping of selected features to their selection iteration scores
        scores_history_: List[float]
            Scores achieved at each iteration
        available_features_: List[str]
            Features that remained unselected
    """
    def __init__(
            self, 
            model: BaseEstimator,
            scorer: callable,
            metric_direction: str = 'maximize',
            cv: Optional[BaseCrossValidator] = None,
            categorical_features: Optional[List[str]] = None,
            min_improvement: float = 1e-4,
            task : str = 'regression',
            include_treatment: bool = False) -> None:
        """
        Initialize the Sequential Forward Selector.
        Parameters:
            model: Any sklearn-compatible estimator
                The machine learning model to use for feature evaluation.
                Can be classifier (binary/multiclass) or regressor.

            scorer: sklearn.metrics._scorer._PredictScorer

                Scoring function (e.g., make_scorer(roc_auc_score))

                Must be compatible with the model's predictions and task type:

                - For binary classification: can use predict_proba (e.g., ROC-AUC)

                - For multiclass: can use predict or predict_proba

                - For regression: uses predict

            metric_direction: str, default='maximize'

                Direction of the optimization ('maximize' or 'minimize')

                - 'maximize' for metrics like accuracy, ROC-AUC

                - 'minimize' for metrics like MSE, MAE

            cv: sklearn cross-validation splitter

                Cross-validation strategy:

                - StratifiedKFold recommended for classification

                - KFold recommended for regression

            categorical_features: List[str], optional

                List of categorical feature names to track during selection

            min_improvement: float, default=1e-4 (0.01%)

                Minimum percentage of improvement required to add a feature (is adapted to score scale)

                1 represents 100% and 0.01 represents 1%

                - For maximize: new_score - base_score must be > min_improvement

                - For minimize: base_score - new_score must be > min_improvement

                It works for all metric scales and it is reresented in percentage.

                - 1 represents 100%

                - 0.1 represents 10%  

                - 0.01 represents 1%

                - 0.001 represents 0.1%

                - 0.0001 represents 0.01%

            task : str, default='regression'

                Defines the task of the dataset we are using.

                Possible values:

                - regression

                - binary_classification

                - multiclass_classification

            

        """
        self.model = model
        self.scorer = scorer
        self.metric_direction = metric_direction
        self.cv = cv
        self.categorical_features = categorical_features or []
        self.supports_categorical = self._check_categorical_support()
        self.min_improvement = min_improvement
        self.task = task
        self.include_treatment = include_treatment


    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> 'SeqForwardSelectorCV':
        """
        Fit the Sequential Forward Selector to the data.
        This method iteratively adds features while monitoring model performance,
        continuing until no feature can be added that improves performance.

        Parameters:
            X: Feature matrix
            y: Target variable (binary, multiclass, or continuous)

        Returns:
            self: The fitted selector
        """
        # Initialize tracking variables
        _, available_features = self._initialize_tracking(X, y)
        iteration = 0
        if self.include_treatment:
            self.selected_features_ = [TREATMENT]
            available_features.remove(TREATMENT)
        while available_features:
            # Get current performance score with selected features
            if len(self.selected_features_) > 0:
                if self.include_treatment:
                    monotone_constraints_list = []
                    for f in self.selected_features_:
                        if f != TREATMENT:
                            monotone_constraints_list.append(0)
                        else:
                            monotone_constraints_list.append(1)
                    self.model.set_params(monotone_constraints=monotone_constraints_list)
              # Only update categorical features if model supports them
                if self.supports_categorical:
                    current_categorical = self._update_categorical_features(self.selected_features_)
                    self.model.set_params(cat_features=current_categorical)
                # Compute the score with the current selected features
                current_score = compute_cv_score(
                    X[self.selected_features_], y, self.model, 
                    self.cv, self.scorer, self.task
                )
            else:
                current_score = float('-inf') if self.metric_direction == 'maximize' else float('inf')

            # Find best performing feature to add
            if self.include_treatment:
                monotone_constraints_list.append(0)
                self.model.set_params(monotone_constraints=monotone_constraints_list)
            best_feature, best_score = self._evaluate_feature_addition(X, y, available_features, current_score)

            # Stop if no feature improves performance
            if best_feature is None:
                print("\nStopping: No features can be added that improve performance")
                break
            # Calculate actual performance gain
            performance_gain = (current_score - best_score if self.metric_direction == 'minimize' else best_score - current_score)

            # Update tracking variables
            self.selected_features_.append(best_feature)
            available_features.remove(best_feature)
            self.feature_scores_[best_feature] = best_score
            self.scores_history_.append(best_score)

            print(f"Iteration {iteration + 1}: Added '{best_feature}' "
                f"(Score: {best_score:.6f}, Performance gain: {performance_gain:.6f})")
            
            iteration += 1

        self.available_features_ = available_features
        self._print_results(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by selecting only the chosen features.
        
        Parameters:
            X: Feature matrix to transform
                
        Returns:
            Transformed feature matrix with selected features only
        """
        return X[self.selected_features_]
    
    def fit_transform(
            self,
            X: pd.DataFrame, 
            y: Union[pd.Series, np.ndarray]
        ) -> pd.DataFrame:
        """
        Fit the selector to the data and transform it in one step.
        Parameters:
            X: Feature matrix
            y: Target variable (binary, multiclass, or continuous)
        Returns:
            Transformed feature matrix with selected features only
        """
        return self.fit(X, y).transform(X)

    def _check_categorical_support(self) -> bool:
        """
        Check if the model supports categorical features.
        Returns:
            bool: True if model supports categorical features, False otherwise
        """
        # First verify if the model has get_params
        if not hasattr(self.model, 'get_params'):
            return False
        # Then verify if cat_features is in the valid parameters
        valid_params = self.model.get_params().keys()
        return 'cat_features' in valid_params
    
    def _update_categorical_features(self, feature_set: List[str]) -> List[str]:
        """
        Update the list of categorical features based on current feature set.
        Only updates if model supports categorical features.

        Parameters:
            feature_set: List of currently selected features
        Returns:
            List of categorical features that are in the current feature set
        """
        if not self.supports_categorical:
            return []
        return [f for f in self.categorical_features if f in feature_set]
    
    def _initialize_tracking(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> tuple:

        """
        Initialize tracking variables.

        

        Parameters:

            X: Feature matrix

            y: Target variable

        

        Returns:

            tuple: (initial_score, list of available features)

        """
        self.selected_features_: List[str] = []
        self.feature_scores_: Dict[str, float] = {}
        self.scores_history_: List[float] = []
        initial_score = float('-inf') if self.metric_direction == 'maximize' else float('inf')
        self.scores_history_.append(initial_score)
        print(f"\nInitial score with no features: {initial_score:.4f}")
        return initial_score, list(X.columns)
    
    def _evaluate_feature_addition(
            self, 
            X: pd.DataFrame, 
            y: Union[pd.Series, np.ndarray], 
            available_features: List[str], 
            base_score: float,
        ) -> tuple:
        """
        Evaluate the impact of adding each available feature and identify the feature 
        that gives the best performance improvement, then check if it meets the threshold.

        Parameters:
            X: Feature matrix
            y: Target variable
            available_features: List of features not yet selected
            base_score: Current base score

        Returns:
            tuple: (feature_to_add, its_score) or (None, None) if no improvement possible
        """
        best_score = float('-inf') if self.metric_direction == 'maximize' else float('inf')
        feature_to_add = None
        best_improvement = float('-inf')
        
        # Special handling for first feature
        is_first_feature = len(self.selected_features_) == 0
        for feature in available_features:
            current_features = self.selected_features_ + [feature]

            if not self.selected_features_ and X[feature].nunique() == 1:
                print(f"Skipping constant feature {feature} as it would be the only feature")
                continue

            try:
                if self.supports_categorical:
                    temp_categorical = self._update_categorical_features(current_features)
                    original_cat_features = self.model.get_params().get('cat_features', [])
                    self.model.set_params(cat_features=temp_categorical)

                score = compute_cv_score(X[current_features], y, self.model, self.cv, self.scorer, self.task)
                
                # Calculate improvement
                if self.metric_direction == 'maximize':
                    improvement = score - base_score
                    if score > best_score:
                        best_score = score
                        feature_to_add = feature
                        best_improvement = improvement

                else:
                    improvement = base_score - score

                    if score < best_score:
                        best_score = score
                        feature_to_add = feature
                        best_improvement = improvement

            except Exception as e:
                print(f"Warning: Evaluation failed for feature {feature}: {str(e)}")
                continue

            finally:
                if self.supports_categorical and original_cat_features is not None:
                    self.model.set_params(cat_features=original_cat_features)

        if not is_first_feature:
            abs_threshold = abs(base_score * self.min_improvement)

            if best_improvement <= abs_threshold:

                return None, None
            
        return feature_to_add, best_score

    def _print_results(self, X: pd.DataFrame) -> None:
        """

        Print the final feature selection results.

        Parameters:

            X: Original feature matrix for counting features

        """
        print(f"\nSFS Feature selection completed:")
        print(f"- Started with 0 features")
        print(f"- Added {len(self.selected_features_)} features")
        print(f"- Left {len(self.available_features_)} features unused")
        print(f"- Final score: {self.scores_history_[-1]:.4f}")
