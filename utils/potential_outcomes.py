import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score, 
    recall_score, 
    precision_score, 
    f1_score,
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    accuracy_score
)
from config import TREATMENT, OUTCOME


def discretize_predictions(pred_probs, threshold):
    """
    Convert predicted probabilities to binary predictions based on the given threshold.

    :param pred_probs: Array of predicted probabilities
    :param threshold: Threshold value for classification
    :return: Array of binary predictions
    """
    return (pred_probs >= threshold).astype(int)


def find_best_threshold(pred_probs, true_values):
    """
    Find the threshold that maximizes the F1 score.

    :param pred_probs: Array of predicted probabilities from the model
    :param true_values: Array of true binary values
    :return: Best threshold value
    """
    best_threshold = 0
    best_f1 = 0

    # Trying thresholds from 0 to 1 at 0.01 intervals
    for threshold in np.arange(0, 1.01, 0.01):
        # Convert probabilities to binary predictions
        preds = (pred_probs >= threshold).astype(int)

        # Compute F1 score
        f1 = f1_score(true_values, preds)

        # Update best threshold if current F1 score is better
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold


def get_metrics(X, y, model, task="classification", model_package="sklearn",):
    
    if (task == "classification") and (model_package == "sklearn"):
        # Compute predictions
        predicted_probs = model.predict_proba(X)[:, 1]
        best_threshold = find_best_threshold(predicted_probs, y)
        preds = discretize_predictions(predicted_probs, best_threshold)

        # Metrics
        print(f"AUC: {roc_auc_score(y, predicted_probs):.2f}")
        print(f"Accuracy: {accuracy_score(y, preds):.2f}")
        print(f"F1: {f1_score(y, preds):.2f}")
        print(f"recall: {recall_score(y, preds):.2f}")
        print(f"precision: {precision_score(y, preds):.2f}")
        
    if (task == "classification") and (model_package == "statsmodels"):
        # Compute predictions
        predicted_probs = model.predict(X)
        best_threshold = find_best_threshold(predicted_probs, y)
        preds = discretize_predictions(predicted_probs, best_threshold)

        # Metrics
        print(f"AUC: {roc_auc_score(y, predicted_probs):.2f}")
        print(f"Accuracy: {accuracy_score(y, preds):.2f}")
        print(f"F1: {f1_score(y, preds):.2f}")
        print(f"recall: {recall_score(y, preds):.2f}")
        print(f"precision: {precision_score(y, preds):.2f}")
    
    if task == "regression":
        # Predictions
        preds = model.predict(X)
        
        # Metrics
        print(f"RMSE: {np.sqrt(mean_squared_error(y, preds)):.2f}")
        print(f"MAE: {mean_absolute_error(y, preds):.2f}")
        print(f"R^2: {r2_score(y, preds):.2f}")


def individual_dose_response_curve(
        df_eval, 
        treatment_interventions,
        predictive_model, 
        modelling_features, 
        feature_counterfactual, 
        scaler=None,
        model_package="sklearn",
        task="classification",
    ):
    """Compute causal effects using the scaled features"""
    potential_outcomes = []
    
    for intervention in treatment_interventions:
        df_interventions = df_eval.copy()
        df_interventions[feature_counterfactual] = intervention
        # Calculate scores
        if (model_package == "sklearn") and (task == "classification"):
            outcome = predictive_model.predict_proba(
                df_interventions[modelling_features].apply(pd.to_numeric)
            )[:, 1]
        elif (model_package == "sklearn") and (task == "regression"):
            outcome = predictive_model.predict(
                df_interventions[modelling_features].apply(pd.to_numeric)
            )
        elif model_package == "statsmodels":
            outcome = predictive_model.predict(
                df_interventions[modelling_features]
            )
        
        potential_outcomes.append(outcome.tolist())
    potential_outcomes = [list(row) for row in zip(*potential_outcomes)]
    return potential_outcomes


def average_dose_response_curve(
        df_eval, 
        treatment_interventions,
        predictive_model, 
        modelling_features, 
        feature_counterfactual, 
        scaler=None,
        model_package="sklearn",
        task="classification",
    ):
    """Compute causal effects using the scaled features"""
    potential_outcomes = []
    
    for intervention in treatment_interventions:
        df_interventions = df_eval.copy()
        df_interventions[feature_counterfactual] = intervention
        # Calculate scores
        if (model_package == "sklearn") and (task == "classification"):
            outcome = predictive_model.predict_proba(
                df_interventions[modelling_features].apply(pd.to_numeric)
            )[:, 1]
        elif (model_package == "sklearn") and (task == "regression"):
            outcome = predictive_model.predict(
                df_interventions[modelling_features].apply(pd.to_numeric)
            )
        elif model_package == "statsmodels":
            outcome = predictive_model.predict(
                df_interventions[modelling_features]
            )
        
        potential_outcomes.append(outcome.mean())
    return potential_outcomes


def individual_real_dose_response_curve(
        df_eval, 
        treatment_interventions,
        generator,
        feature_counterfactual
    ):
    """Compute causal effects using the scaled features"""
    real_outcomes = []
    
    for intervention in treatment_interventions:
        df_interventions = df_eval.copy()
        df_interventions[feature_counterfactual] = intervention
        outcome = generator.calculate_outcome_probability(df_interventions, df_interventions[feature_counterfactual])
        
        real_outcomes.append(outcome.tolist())
    real_outcomes = [list(row) for row in zip(*real_outcomes)]
    return real_outcomes


def plot_score_trend(data):
    grouped_data = data.groupby(TREATMENT).mean()["score"].reset_index()
    slope, intercept = np.polyfit(grouped_data[TREATMENT], grouped_data["score"], 1)
    print(F"Slope: {slope:.5f}, (ATE: {slope*100:.2f})")
    print(F"Intercept: {intercept:.5f}")
    f, ax = plt.subplots(figsize=(8, 7))
    sns.lineplot(data=grouped_data, x=TREATMENT, y="score")
    plt.ylim(0, 1)
    plt.show()
    