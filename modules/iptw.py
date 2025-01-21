import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from config import TREATMENT, OUTCOME
from utils.gps import create_sample_weights


class ContinuousIPTW:
    def __init__(self, model, n_folds=5, random_state=42):
        """
        Initialize the Continuous IPTW calculator
        
        Parameters:
        -----------
        ml_model : estimator object, optional
            The machine learning model to use for propensity score estimation.
            Must implement fit and predict methods.
            Default is GradientBoostingRegressor
        n_folds : int, optional
            Number of folds for cross-validation
        random_state : int, optional
            Random state for reproducibility
        """
        self.model = model
        self.n_folds = n_folds
        self.random_state = random_state

    def fit_model(self, X, t, sample_weights=None):
        model_propensity = clone(self.model)
        model_propensity.fit(X, t, sample_weight=sample_weights)
        return model_propensity
        
    def fit_predict(self, X, treatment, bin_edges):
        """
        Fit the model and predict propensity scores using cross-validation
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The covariate matrix
        treatment : array-like of shape (n_samples,)
            The continuous treatment variable
            
        Returns:
        --------
        array-like
            Predicted propensity scores for each sample
        """
        # Data
        data_copy = X.copy()
        data_copy[TREATMENT] = treatment
        data_copy["treatment_predictions_oos"] = 0.0
        treatment_values = np.array(data_copy[TREATMENT].to_numpy()).flatten()
        treatment_bins = pd.cut(data_copy[TREATMENT], bins=bin_edges, labels=False)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Perform cross-validation
        for train_idx, test_idx in kf.split(X, treatment_bins):
            X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
            t_train, t_test = treatment.iloc[train_idx, :][[TREATMENT]], treatment.iloc[test_idx, :][[TREATMENT]]
            sample_weight_fold = create_sample_weights(t_train, bin_edges)

            model_t_fold = self.fit_model(X_train, t_train, sample_weight_fold)

            # Predict on test data
            data_copy.iloc[test_idx, -1] = model_t_fold.predict(X_test)
        
        # refit model on all data for OOS inferences
        sample_weights = create_sample_weights(treatment, bin_edges)
        self.model_t = self.fit_model(X, treatment, sample_weights)
            
        return data_copy["treatment_predictions_oos"]
    
    def compute_weights(self, X, treatment, bin_edges):
        """
        Compute inverse probability weights for continuous treatment
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The covariate matrix
        treatment : array-like of shape (n_samples,)
            The continuous treatment variable
        bandwidth : float, optional
            Bandwidth for kernel density estimation.
            If None, Silverman's rule of thumb is used.
            
        Returns:
        --------
        array-like
            Inverse probability weights for each sample
        """
        # Get predicted propensity scores
        pred_treatment = self.fit_predict(X, treatment, bin_edges)
        
        # data
        data_copy = X.copy()
        data_copy[TREATMENT] = treatment
        data_copy["treatment_predictions_oos"] = pred_treatment
        
        # Extract residuaks and residuals variance
        data_copy["treatment_residuals_oos"] = data_copy[TREATMENT] - data_copy["treatment_predictions_oos"] 
        sigma_hat = np.sqrt(np.mean(data_copy["treatment_residuals_oos"]**2))  # MSE -> sigma^2t)

        # get best KDE params
        residuals = data_copy["treatment_residuals_oos"].to_numpy().reshape(-1, 1)
        treatment_values = np.array(data_copy[TREATMENT].to_numpy()).flatten()
        best_params = self.optimize_kde_params(residuals)

        # Fit KDE
        self.kde = KernelDensity(**best_params)
        self.kde.fit(residuals)
        gps_values = np.exp(self.kde.score_samples(residuals))
        data_copy["treatment_propensity_score"] = gps_values
        
        print("\nFirst 5 GPS values:\n", gps_values[:5])
        print("\nEstimated RMSE:", sigma_hat)
        print(f"Estimated MAE: {np.mean(data_copy['treatment_residuals_oos'].abs())}")
        
        # Compute inverse weights
        epsilon = 0.01 # Choose an appropriate small value
        gps_values_safe = np.where(gps_values == 0, epsilon, gps_values)
        weights = 1 / gps_values_safe
        
        # Normalize weights
        weights = weights / np.mean(weights)
        
        return weights

    def compute_oos_weights(self, X, treatment):
        pred_treatment = self.model_t.predict(X)
        
        # data
        data_copy = X.copy()
        data_copy["treatment_predictions_oos"] = pred_treatment
        data_copy[TREATMENT] = treatment

        
        # Extract residuals
        data_copy["treatment_residuals_oos"] = data_copy[TREATMENT] - data_copy["treatment_predictions_oos"] 
        residuals = data_copy["treatment_residuals_oos"].to_numpy().reshape(-1, 1)
        treatment_values = np.array(data_copy[TREATMENT].to_numpy()).flatten()

        # Fit KDE
        gps_values = np.exp(self.kde.score_samples(residuals))
        data_copy["treatment_propensity_score"] = gps_values
        print("\nFirst 5 GPS values:\n", gps_values[:5])
        
        # Compute inverse weights
        epsilon = 0.001  # Choose an appropriate small value
        gps_values_safe = np.where(gps_values < epsilon, epsilon, gps_values)
        weights = 1 / gps_values_safe
        
        # Normalize weights
        weights = weights / np.mean(weights)
        
        return weights
    
    def optimize_kde_params(self, X):
        # Define parameter grid
        param_grid = {
            'bandwidth': np.logspace(-1, 1, 20),  # Test bandwidths from 0.1 to 10
            # 'bandwidth': np.linspace(1e-3, 1, 30),
            'kernel': ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
        }

        # Initialize and fit GridSearchCV
        kde = KernelDensity()
        grid_search = GridSearchCV(
            kde, 
            param_grid=param_grid,
            cv=5,  
            n_jobs=-1  # Use all available cores
        )
        grid_search.fit(X)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_}")

        return grid_search.best_params_