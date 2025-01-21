import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import make_pipeline
from collections import defaultdict
from config import TREATMENT


def optimize_kde_params(X):
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


def plot_kde_vs_hist(residuals, gps_values, kde_model):
    plt.figure(figsize=(12, 6))
    
    # Plot histogram of actual data, normalized to form a density
    plt.hist(residuals, bins=30, density=True, alpha=0.6, color='skyblue', label='Actual Distribution')
    
    # # Plot KDE density
    kde_density = np.exp(kde_model.score_samples(residuals))
    plt.scatter(residuals, gps_values, alpha=0.5, s=1, c='red', label='KDE Estimate (GPS values)')

    
    plt.title('Comparison of Actual Distribution vs KDE Estimate')
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_gps_diagnostics(treatment_values, gps_values):
    plt.figure(figsize=(15, 5))
    
    # Plot 1: GPS distribution
    plt.subplot(131)
    plt.hist(gps_values, bins=50, density=True)
    plt.title('GPS Distribution')
    plt.xlabel('GPS Values')
    plt.ylabel('Density')
    plt.xticks(rotation=45)
    
    # Plot 2: GPS vs Treatment
    plt.subplot(132)
    plt.scatter(treatment_values, gps_values, alpha=0.5)
    plt.title('GPS vs Treatment')
    plt.xlabel('Treatment')
    plt.ylabel('GPS')
    plt.xticks(rotation=45)
    
    # Plot 3: GPS percentiles across treatment levels
    plt.subplot(133)
    treatment_bins = pd.qcut(treatment_values, q=10)
    gps_by_treatment = pd.DataFrame({
        'treatment_bin': treatment_bins,
        'gps': gps_values
    }).boxplot(column='gps', by='treatment_bin')
    plt.title('GPS Distribution by Treatment Decile')
    plt.xlabel('Treatment Decile')
    plt.ylabel('GPS')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()


def compute_interval_probabilities_kde(predictions, kde_model, intervals):
    """
    Given per-customer mu_hat and a fitted KDE model,
    compute the probability of T falling into each interval
    for every customer.
    
    Parameters:
    mu_hat: predictions from the model (treatment_predictions_oos)
    kde_model: fitted KernelDensity object
    intervals: list of (a, b) for each interval (a, b]
    
    Returns: 2D numpy array of shape (n_customers, n_intervals)
    """
    n_customers = len(predictions)
    n_intervals = len(intervals)
    probs = np.zeros((n_customers, n_intervals))
    
    # Number of points for numerical integration
    n_points = 100
    
    for i in range(n_customers):
        if i % 100 == 0:
            print(i)
        for j, (a, b) in enumerate(intervals):
            # Create points for numerical integration in current interval
            points = np.linspace(a - predictions[i], b - predictions[i], n_points).reshape(-1, 1)
            
            # Get log density at these points
            log_density = kde_model.score_samples(points)
            
            # Convert to density and compute integral using trapezoidal rule
            density = np.exp(log_density)
            prob = np.trapz(density, points.ravel())
            
            probs[i, j] = prob
        
        # Normalize probabilities for this sample to sum to 1
        # row_sum = probs[i, :].sum()
        # if (row_sum > 0) and (row_sum < 1) :  # Avoid division by zero
            # probs[i, :] = probs[i, :] / row_sum
            
    return probs


def find_samples_by_threshold(interval_probs, row_ids, intervals, threshold, symbol):
    """
    Given:
      - interval_probs: 2D numpy array of shape (n_samples, n_intervals)
      - intervals: list of (a, b) tuples representing intervals, length = n_intervals
      - threshold: float

    Returns:
      A dictionary where each key is the sample index (0-based),
      and the value is a list of intervals that have probability < threshold.
    """
    # Dictionary to store the results
    results = {}

    # Iterate over each sample (row in interval_probs)
    for row_id, i in zip(row_ids, range(interval_probs.shape[0])):
        # Find which intervals are below the threshold
        below_mask = interval_probs[i] < threshold if symbol == "<" else interval_probs[i] >= threshold
        # Get the indices of those intervals
        below_indices = np.where(below_mask)[0]

        if below_indices.size > 0:
            # Map those indices back to the actual interval tuples
            intervals_below = [intervals[j] for j in below_indices]
            # Store in the dictionary
            results[row_id] = intervals_below

    return results


def create_sample_weights(df_split, bin_cuts):
    treatment_bins_sample = pd.cut(df_split[TREATMENT], bins=bin_cuts, labels=False,)
    class_counts = treatment_bins_sample.value_counts()
    n_samples = len(treatment_bins_sample)
    n_classes = len(class_counts)

    # Calculate balanced weights (inverse of frequency)
    class_weights = n_samples / (n_classes * class_counts)

    # Map these weights to each sample based on its class
    sample_weights = treatment_bins_sample.map(class_weights)

    # Normalize weights to sum to n_samples (optional but recommended)
    sample_weights = sample_weights * (n_samples / sample_weights.sum())
    return sample_weights