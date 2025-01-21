import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats

class PartialCorrelationFeatureSelector:
    """
    Feature selector that removes highly correlated features and features that don't 
    substantially change the correlation between treatment and outcome when controlled for.
    
    Parameters
    ----------
    correlation_threshold : float, default=0.8
        Features with correlation coefficient higher than this value will be considered
        for removal.
    
    treatment_variable : str
        Name of the treatment variable.
        
    min_partial_correlation : float, default=None
        Minimum absolute difference required between the baseline treatment-outcome correlation
        and the partial correlation when controlling for a feature. Features that cause
        a smaller change than this threshold will be removed. If None, this filter is not applied.
    
    Attributes
    ----------
    selected_features_ : list
        Features selected after fitting.
    feature_partial_correlations_ : dict
        Dictionary containing the partial correlation values between treatment and outcome
        when controlling for each feature.
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
            First variable (treatment)
        y : array-like
            Second variable (outcome)
        z : array-like
            Control variable (feature)
            
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
        
        # Get treatment variable values
        treatment = X[self.treatment_variable].values
        
        # Compute correlation matrix for features only (excluding treatment)
        corr_matrix = X[features].corr().abs()
        
        # Get pairs of features with correlation above threshold
        high_corr_pairs = []
        for i, j in combinations(range(len(features)), 2):
            if corr_matrix.iloc[i, j] > self.correlation_threshold:
                high_corr_pairs.append((features[i], features[j]))
        
        # First compute baseline correlation between treatment and outcome
        baseline_corr = stats.pearsonr(treatment, y.values)[0]
        
        # Compute partial correlations between treatment and outcome controlling for each feature
        # and store the difference from baseline correlation
        self.feature_partial_correlations_ = {}
        for feat in features:
            partial_corr = self._compute_partial_correlation(
                treatment,  # treatment (x)
                y.values,   # outcome (y)
                X[feat].values  # controlling feature (z)
            )
            # Store the absolute difference between partial and baseline correlation
            # A larger difference means the feature has more impact on the treatment-outcome relationship
            correlation_difference = abs(partial_corr - baseline_corr)
            self.feature_partial_correlations_[feat] = correlation_difference
            # print(feat, correlation_difference)
        
        # First, remove features based on the correlation difference threshold
        features_to_remove = set()
        if self.min_partial_correlation is not None:
            for feat, corr_diff in self.feature_partial_correlations_.items():
                # Remove features where the correlation difference is LESS than the threshold
                if corr_diff < self.min_partial_correlation:
                    features_to_remove.add(feat)
        
        # Then, for each highly correlated pair, keep the feature that results in
        # lower partial correlation between treatment and outcome
        for feat1, feat2 in high_corr_pairs:
            # Skip if either feature was already removed
            if feat1 in features_to_remove or feat2 in features_to_remove:
                continue
                
            partial_corr1 = abs(self.feature_partial_correlations_[feat1])
            partial_corr2 = abs(self.feature_partial_correlations_[feat2])
            
            # Remove feature that results in higher partial correlation
            if partial_corr1 > partial_corr2:
                features_to_remove.add(feat2)
            else:
                features_to_remove.add(feat1)
        
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