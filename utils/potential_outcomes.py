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
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier, CatBoostRegressor

from config import TREATMENT, OUTCOME, SEED
from utils.estimates import logistic_regression_estimation, s_learner_estimation, augmented_iptw_estimation


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

    if (task == "classification") and (model_package == "self_aiptw"):
        # Compute predictions
        predicted_probs = model.predict(X)
        predicted_probs = np.clip(predicted_probs, 0, 1)
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
        elif model_package == "self_aiptw":
            outcome = predictive_model.predict(
                df_interventions[modelling_features]
            )
            outcome = np.clip(outcome, 0, 1)
        elif model_package == "self_aiptw_calibrated":
            outcome_uncalib = predictive_model[0].predict(
                df_interventions[modelling_features]
            )
            df_interventions_copy = df_interventions.copy()
            df_interventions_copy["final_model_oos_predictions"] = outcome_uncalib
            outcome = predictive_model[1].predict_proba(
                df_interventions_copy[["final_model_oos_predictions"]]
            )[:, 1]
            
        
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
        elif model_package == "self_aiptw":
            outcome = predictive_model.predict(
                df_interventions[modelling_features]
            )
            outcome = np.clip(outcome, 0, 1)
        elif model_package == "self_aiptw_calibrated":
            outcome_uncalib = predictive_model[0].predict(
                df_interventions[modelling_features]
            )
            df_interventions_copy = df_interventions.copy()
            df_interventions_copy["final_model_oos_predictions"] = outcome_uncalib
            outcome = predictive_model[1].predict_proba(
                df_interventions_copy[["final_model_oos_predictions"]]
            )[:, 1]
        
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
    
    original_treatments = df_eval[TREATMENT].to_numpy().flatten()
    original_outcomes = df_eval[OUTCOME].to_numpy().flatten()
    for intervention in treatment_interventions:
        df_interventions = df_eval.copy()
        df_interventions[feature_counterfactual] = intervention
        probs, _ = generator.calculate_outcome_probability(df_interventions, df_interventions[feature_counterfactual])

        outcomes = []
        for prob, outcome, original_t in zip(probs, original_outcomes, original_treatments):
            if (outcome == 1) & (intervention >= original_t):
                outcomes.append(1)
            elif (outcome == 0) & (intervention <= original_t):
                outcomes.append(0)
            else:
                outcomes.append(prob)
        
        real_outcomes.append(outcomes)
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



def get_ci_estimates(
    estimation_method,
    train_df,
    n_iterations,
    intervention_df,
    intervention_values,
    adjustment_set,
    outcome_causes,
    confidence_level=0.95,
    monotone_constrains=None,
):
    average_dose_response_curves = []

    for i in range(n_iterations):
        # Create a bootstrapped dataset
        features_model = adjustment_set + [TREATMENT]
        train_df_resampled = train_df.sample(n=len(train_df), replace=True, random_state=SEED+i)
        X_resampled, y_resampled = (
            train_df_resampled[features_model],
            train_df_resampled[[OUTCOME]],
        )

        # train model and predict POs
        if estimation_method == "logistic_reg":
            if len(outcome_causes) > 0:
                binned_outcome_causes = [c  + '_bins' for c in outcome_causes]
                for outcome_cause in outcome_causes:
                    bins = pd.qcut(train_df_resampled[outcome_cause], q=5, retbins=True, duplicates='drop')[1]
                    max_val_train = train_df_resampled[outcome_cause].max()
                    max_val_int = intervention_df[outcome_cause].max() 
                    bins[-1] = max_val_train if max_val_train > max_val_int else max_val_int
                    min_val_train = train_df_resampled[outcome_cause].min()
                    min_val_int = intervention_df[outcome_cause].min()
                    bins[0] = min_val_train if min_val_train < min_val_int else min_val_int
                    train_df_resampled[outcome_cause + '_bins'] = pd.cut(
                        train_df_resampled[outcome_cause], bins=bins, include_lowest=True
                    )
                    intervention_df[outcome_cause + '_bins'] = pd.cut(
                        intervention_df[outcome_cause], bins=bins, include_lowest=True
                    )
            else:
                binned_outcome_causes = []
            log_regression_boostrap = logistic_regression_estimation(
                train_df_resampled, adjustment_set, binned_outcome_causes
            )
            individual_potential_outcome = individual_dose_response_curve(
                df_eval=intervention_df,
                treatment_interventions=intervention_values,
                predictive_model=log_regression_boostrap, 
                modelling_features=adjustment_set + binned_outcome_causes + [TREATMENT], 
                feature_counterfactual=TREATMENT, 
                model_package="statsmodels",
                task="classification"
            )
            # print(individual_potential_outcome[:5])

        elif estimation_method == "s_learner":
            s_learner_params = {
                "n_estimators": 500,
                "depth": None,
                "min_data_in_leaf": round(X_resampled.shape[0]*0.05),
                "learning_rate": 0.01,
                "subsample": 1,
                "rsm": 1,
                "objective": "Logloss",
                "silent": True,
                "l2_leaf_reg": 0,
                "random_seed": SEED
            }
            s_learner_params["monotone_constraints"] = monotone_constrains
            s_learner_model = CatBoostClassifier(**s_learner_params)
            s_learner_boostrap = s_learner_estimation(X_resampled, y_resampled, features_model, s_learner_model)
            individual_potential_outcome = individual_dose_response_curve(
                df_eval=intervention_df,
                treatment_interventions=intervention_values,
                predictive_model=s_learner_boostrap, 
                modelling_features=features_model, 
                feature_counterfactual=TREATMENT, 
                model_package="sklearn",
                task="classification"
            )

        elif estimation_method == "aiptw":
            iptw_controls = [c for c in adjustment_set if c not in outcome_causes]
            propensity_params = {
                "n_estimators": 200,
                "depth": None,
                "min_data_in_leaf": round(X_resampled.shape[0]*(2/3)*0.01),
                "learning_rate": 0.01,
                "subsample": 1,
                "rsm": 1,
                "objective": "RMSE",
                "silent": True,
                "l2_leaf_reg": 3,
                "random_seed": SEED
            }
            model_propensity = CatBoostRegressor(**propensity_params)
            bin_edges_contained = intervention_values
            bin_edges_contained[0] = -1


            outcome_params = {
                "n_estimators": 500,
                "depth": None,
                "min_data_in_leaf": round(X_resampled.shape[0]*(2/3)*0.05),
                "learning_rate": 0.01,
                "subsample": 1,
                "rsm": 1,
                "objective": "Logloss",
                "silent": True,
                "l2_leaf_reg": 0,
                "random_seed": SEED
            }
            outcome_params["monotone_constraints"] = monotone_constrains
            outcome_model = CatBoostClassifier(**outcome_params)

            final_model_params = {
                "n_estimators": 500,
                "depth": None,
                "min_data_in_leaf": round(X_resampled.shape[0]*0.05),
                "learning_rate": 0.01,
                "subsample": 1,
                "rsm": 1,
                "objective": "RMSE",
                "silent": True,
                "l2_leaf_reg": 0,
                "random_seed": SEED,
            }
            final_model_params["monotone_constraints"] = monotone_constrains
            final_model_aiptw = CatBoostRegressor(**final_model_params)

            calibration_model_aiptw = LogisticRegression(solver="lbfgs")

            final_model_aiptw_boostrap, final_model_aiptw_boostrap_calib = augmented_iptw_estimation(
                X_train_obs=X_resampled, 
                y_train_obs=y_resampled, 
                cv_folds=3,
                bin_edges=bin_edges_contained,
                features=features_model,
                gps_controls=iptw_controls, 
                propensity_model=model_propensity, 
                outcome_model=outcome_model, 
                final_model=final_model_aiptw,
                model_calibration=calibration_model_aiptw,
            )
            individual_potential_outcome = individual_dose_response_curve(
                df_eval=intervention_df,
                treatment_interventions=intervention_values,
                predictive_model=[final_model_aiptw_boostrap, final_model_aiptw_boostrap_calib], 
                modelling_features=features_model, 
                feature_counterfactual=TREATMENT, 
                model_package="self_aiptw_calibrated",
                task="classification"
            )

        individual_potential_outcomes_resample = np.array(individual_potential_outcome)
        average_dose_response_curve_resample = np.mean(individual_potential_outcomes_resample, axis=0)
        average_dose_response_curves.append(average_dose_response_curve_resample)

    # Convert predictions to a NumPy array
    average_dose_response_curves = np.array(average_dose_response_curves)

    # Average
    average_dose_response_curve = np.mean(average_dose_response_curves, axis=0)

    # Confidence intervals
    lower_bound_dose_response = np.percentile(average_dose_response_curves, (1 - confidence_level) * 100, axis=0)
    upper_bound_dose_response = np.percentile(average_dose_response_curves, confidence_level * 100, axis=0)

    results = {
        "dose-response": {
            "average": average_dose_response_curve,
            "lower_bound": lower_bound_dose_response,
            "upper_bound": upper_bound_dose_response
        },

    }
    return results