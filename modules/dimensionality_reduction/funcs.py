import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


def get_predictions(model, X, task, classes=None):
    """
    Get predictions using the appropriate prediction method.
    
    Parameters:
        model: fitted sklearn-compatible estimator
        X: Feature matrix
        prediction_type: str, type of prediction needed
        classes: array-like, optional, unique classes for multiclass problems
        
    Returns:
        Model predictions in the appropriate format
    """
    if task == 'multiclass_classification':
        # Get probability predictions for each class
        return model.predict_proba(X)
    elif task == 'binary_classification':
        # For binary classification, return probability of positive class
        return model.predict_proba(X)[:, 1]
    else:  # 'predict' or 'regression'
        return model.predict(X)

def check_classification_targets(y):
    """
    Check if target variable has enough classes in the dataset.
    
    Parameters:
        y: Target variable
        
    Returns:
        bool: True if targets are valid for classification
    """
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError("Target variable must have at least two unique classes for classification.")
    return True

def compute_cv_score(X, y, model, cv, scorer, task):
    """
    Compute the cross-validated score for the current feature set.
    
    This method handles all types of prediction tasks (binary classification,
    multiclass classification, and regression) by using the appropriate
    prediction method based on the detected prediction type.
    
    Parameters:
        X: Feature matrix with current set of features
        y: Target variable (binary, multiclass, or continuous)
        model: sklearn-compatible estimator
        cv: cross-validation splitter
        scorer: scoring function
        prediction_type: str, type of prediction needed
            
    Returns:
        float: Mean cross-validated score across all folds
    """
    scores = []
    
    # For classification, verify we have enough classes
    if task in ['binary_classification', 'multiclass_classification']:
        check_classification_targets(y)
        classes = np.unique(y)
    else:
        classes = None
    
    # Iterate through cross-validation folds
    for train_idx, val_idx in cv.split(X, y):
        # Split data into training and validation sets for this fold
        if isinstance(X, pd.DataFrame):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        else:
            X_train, X_val = X[train_idx], X[val_idx]
            
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        else:
            y_train, y_val = y[train_idx], y[val_idx]
        
        # Verify we have all classes in both training and validation sets
        if task in ['binary_classification', 'multiclass_classification']:
            train_classes = np.unique(y_train)
            val_classes = np.unique(y_val)
            if len(train_classes) < len(classes) or len(val_classes) < len(classes):
                continue  # Skip this fold if not all classes are present
        
        # Train the model on the training data
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        try:
            y_pred = get_predictions(model_clone, X_val, task, classes)
            
            if task == 'multiclass_classification' and 'average_precision_score' in str(scorer._score_func):
                y_val_bin = label_binarize(y_val, classes=classes)
                score = scorer._score_func(y_val_bin, y_pred, **scorer._kwargs)
            else:
                score = scorer._score_func(y_val, y_pred, **scorer._kwargs)
                
            scores.append(score)
            
        except ValueError as e:
            print(f"Warning: Scoring error in fold: {str(e)}")
            continue
    
    if not scores:
        raise ValueError("No valid folds found for scoring. Ensure your dataset has enough samples of each class.")
    
    return np.mean(scores)


def kmeans_discretize(X, num_bins=5):
    """
    Discretize numeric features using k-means clustering.
    Only discretizes continuous numeric columns.
    
    Parameters:
    -----------
    X : pd.DataFrame or pd.Series
        Input data
    num_bins : int, default=5
        Number of bins
        
    Returns:
    --------
    pd.DataFrame or pd.Series
        Discretized data
    """
    def _kmeans_bin(series):
        kbd = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='kmeans', subsample=None)
        return pd.Series(
            kbd.fit_transform(series.values.reshape(-1, 1)).flatten() + 1,
            index=series.index
        )
    
    # Handle Series input
    if isinstance(X, pd.Series):
        return _kmeans_bin(X) if len(X.unique()) > 10 else X
        
    # Handle DataFrame input    
    X_disc = X.copy()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    for col in num_cols:
        if len(X[col].unique()) > 10:
            X_disc[col] = _kmeans_bin(X[col])
            
    return X_disc

class ImputeMissing(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='median'):
        self.strategy = strategy
        self.imputer = SimpleImputer(missing_values=np.nan, strategy=strategy, keep_empty_features=True)
        self.feature_names = None  # To store feature names for later

    def fit(self, X, y=None):
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.imputer.fit(X[numeric_features], y)
        self.numerical_feature_names = numeric_features 
        return self

    def transform(self, X):
        # Apply the imputer and then convert back to DataFrame with original column names
        X[self.numerical_feature_names] = self.imputer.transform(X[self.numerical_feature_names])
        return X
    
def encode_categorical_features(data):
    """
    Encode categorical features in a DataFrame or Series using LabelEncoder.
    Handles both DataFrame and Series inputs, preserving the input type in the output.
    Does not handle missing values - data should be preprocessed beforehand.
    
    Parameters:
    -----------
    data : pandas.DataFrame or pandas.Series
        Input data containing categorical features
        
    Returns:
    --------
    pandas.DataFrame or pandas.Series
        Data with categorical features encoded, maintaining the same type as input
    """
    # Handle Series input
    if isinstance(data, pd.Series):
        # Only encode if the Series is categorical or object type
        if data.dtype in ['object', 'category']:
            encoder = LabelEncoder()
            return pd.Series(encoder.fit_transform(data), index=data.index)
        return data
    
    # Handle DataFrame input
    df_encoded = data.copy()
    
    # Get categorical columns (object and category dtypes)
    categorical_columns = df_encoded.select_dtypes(include=['object', 'category']).columns
    
    # If no categorical columns, return original df
    if len(categorical_columns) == 0:
        return df_encoded
    
    # Apply LabelEncoder to each categorical column
    for column in categorical_columns:
        encoder = LabelEncoder()
        df_encoded[column] = encoder.fit_transform(df_encoded[column])
    
    return df_encoded