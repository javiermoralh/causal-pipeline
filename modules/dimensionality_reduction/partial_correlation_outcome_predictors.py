import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats

class OutcomePartialCorrelationFeatureSelector:
    """
    Feature selector that removes highly correlated features and features with low
    partial correlation with target controlling for treatment variable.
    
    Parameters
    ----------
    correlation_threshold : float, default=0.8
        Features with correlation coefficient higher than this value will be considered
        for removal.
    
    treatment_variable : str
        Name of the treatment variable to control for in partial correlation analysis.
        
    min_partial_correlation : float, default=None
        Minimum absolute partial correlation required between a feature and the outcome
        (controlling for treatment) for the feature to be retained. If None, this filter
        is not applied.
    
    Attributes
    ----------
    selected_features_ : list
        Features selected after fitting.
    feature_partial_correlations_ : dict
        Dictionary containing the partial correlation values for each feature with the outcome.
    """
    
    def __init__(self, correlation_threshold=0.8, treatment_variable=None, min_partial_correlation=None):
        self.correlation_threshold = correlation_threshold
        self.treatment_variable = treatment_variable
        self.min_partial_correlation = min_partial_correlation
        self.selected_features_ = None
        self.feature_partial_correlations_ = {}
    
    def _compute_partial_correlation(self, x, y, z):
        """
        Compute partial correlation between x and y controlling for z.
        
        Parameters
        ----------
        x : array-like
            First variable
        y : array-like
            Second variable
        z : array-like
            Control variable
            
        Returns
        -------
        float
            Partial correlation coefficient
        """
        # Compute residuals of x ~ z
        res_x = stats.linregress(z, x)[0] * z + stats.linregress(z, x)[1] - x
        
        # Compute residuals of y ~ z
        res_y = stats.linregress(z, y)[0] * z + stats.linregress(z, y)[1] - y
        
        # Compute correlation between residuals
        return stats.pearsonr(res_x, res_y)[0]
    
    def fit(self, X, y):
        """
        Fit the feature selector.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series or pandas.DataFrame
            Target variable
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Convert y to series if it's a dataframe
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        
        # Get feature names excluding treatment variable
        features = [col for col in X.columns if col != self.treatment_variable]
        
        # Compute correlation matrix
        corr_matrix = X[features].corr().abs()
        
        # Get pairs of features with correlation above threshold
        high_corr_pairs = []
        for i, j in combinations(range(len(features)), 2):
            if corr_matrix.iloc[i, j] > self.correlation_threshold:
                high_corr_pairs.append((features[i], features[j]))
        
        # Get treatment variable values
        z = X[self.treatment_variable].values
        
        # Compute partial correlations with outcome for all features
        self.feature_partial_correlations_ = {}
        for feat in features:
            partial_corr = self._compute_partial_correlation(
                X[feat].values, y.values, z
            )
            self.feature_partial_correlations_[feat] = partial_corr
        
        # First, remove features with low partial correlation if threshold is set
        features_to_remove = set()
        if self.min_partial_correlation is not None:
            for feat, corr in self.feature_partial_correlations_.items():
                if abs(corr) < self.min_partial_correlation:
                    features_to_remove.add(feat)
        
        # Then, for each highly correlated pair, remove feature with lower partial correlation
        for feat1, feat2 in high_corr_pairs:
            # Skip if either feature was already removed due to low correlation
            if feat1 in features_to_remove or feat2 in features_to_remove:
                continue
                
            partial_corr1 = abs(self.feature_partial_correlations_[feat1])
            partial_corr2 = abs(self.feature_partial_correlations_[feat2])
            
            # Remove feature with lower partial correlation
            if partial_corr1 < partial_corr2:
                features_to_remove.add(feat1)
            else:
                features_to_remove.add(feat2)
        
        # Create list of selected features
        self.selected_features_ = [
            feat for feat in X.columns 
            if feat not in features_to_remove
        ]
        
        return self
    
    def transform(self, X):
        """
        Transform the data by selecting only the chosen features.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix
            
        Returns
        -------
        pandas.DataFrame
            Transformed feature matrix
        """
        if self.selected_features_ is None:
            raise ValueError("Selector has not been fitted yet. Call 'fit' first.")
        
        return X[self.selected_features_]
    
    def fit_transform(self, X, y):
        """
        Fit the selector and transform the data.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series or pandas.DataFrame
            Target variable
            
        Returns
        -------
        pandas.DataFrame
            Transformed feature matrix
        """
        return self.fit(X, y).transform(X)