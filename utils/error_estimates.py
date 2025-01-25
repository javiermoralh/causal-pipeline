import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from sklearn.base import clone
from catboost import CatBoostClassifier, CatBoostRegressor

from config import TREATMENT, OUTCOME
from utils.estimates import logistic_regression_estimation, s_learner_estimation, iptw_estimation
from utils.potential_outcomes import individual_dose_response_curve, individual_real_dose_response_curve


def compute_arrays_estimation_rmse(pred, label):
    # Convert to numpy arrays if they aren't already
    pred = np.array(pred)
    label = np.array(label)
    
    # Calculate RMSE for each column (intervention)
    rmse_list = []
    for col in range(pred.shape[1]):
        # Get corresponding columns from pred and label
        pred_col = pred[:, col]
        label_col = label[:, col]
        
        # Remove any entries where pred has missing values (0.)
        mask = pred_col != 0.
        pred_col = pred_col[mask]
        label_col = label_col[mask]
        
        # Calculate RMSE for this column
        mse = np.mean((pred_col - label_col) ** 2)
        rmse = np.sqrt(mse)
        rmse_list.append(rmse)
    
    return rmse_list


def get_ci_estimation_results(
    estimation_method,
    train_df,
    n_iterations,
    intervention_df,
    intervention_values,
    adjustment_set,
    outcome_causes,
    generator,
    confidence_level=0.95,
    monotone_constrains=None,
):
    average_dose_response_curves = []
    errors_estimation = []

    for i in range(n_iterations):
        # Create a bootstrapped dataset
        features_model = adjustment_set + outcome_causes + [TREATMENT]
        train_df_resampled = train_df.sample(n=len(train_df), replace=True)
        X_resampled, y_resampled = (
            train_df_resampled[features_model],
            train_df_resampled[[OUTCOME]],
        )

        # train model and predict POs
        if estimation_method == "logistic_reg":
            if len(outcome_causes) > 0:
                binned_outcome_causes = [c  + '_bins' for c in outcome_causes]
                for outcome_cause in outcome_causes:
                    bins = pd.qcut(train_df[outcome_cause], q=5, retbins=True, duplicates='drop')[1]
                    train_df_resampled[outcome_cause + '_bins'] = pd.cut(train_df_resampled[outcome_cause], bins=bins, include_lowest=True)
                    intervention_df[outcome_cause + '_bins'] = pd.cut(intervention_df[outcome_cause], bins=bins, include_lowest=True)
            else:
                binned_outcome_causes = []
            log_regression_boostrap = logistic_regression_estimation(train_df_resampled, adjustment_set, binned_outcome_causes)
            individual_potential_outcome = individual_dose_response_curve(
                df_eval=intervention_df,
                treatment_interventions=intervention_values,
                predictive_model=log_regression_boostrap, 
                modelling_features=adjustment_set + binned_outcome_causes + [TREATMENT], 
                feature_counterfactual=TREATMENT, 
                model_package="statsmodels",
                task="classification"
            )
            # print(individual_potential_outcome)

        elif estimation_method == "s_learner":
            s_learner_params = {
                "n_estimators": 200,
                "depth": None,
                "min_data_in_leaf": round(X_resampled.shape[0]*0.01),
                "learning_rate": 0.01,
                "subsample": 1,
                "rsm": 1,
                "objective": "Logloss",
                "silent": True,
                "l2_leaf_reg": 3
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

        elif estimation_method == "iptw":
            init_params = {
                "n_estimators": 200,
                "depth": None,
                "min_data_in_leaf": round(X_resampled.shape[0]*(2/3)*0.01),
                "learning_rate": 0.01,
                "subsample": 1,
                "rsm": 1,
                "objective": "RMSE",
                "silent": True,
                "l2_leaf_reg": 3
            }
            model_propensity = CatBoostRegressor(**init_params)
            bin_edges_contained = intervention_values
            bin_edges_contained[0] = -1
            weighter = ContinuousIPTW(model=model_propensity, n_folds=5, random_state=42)
            weights_iptw = weighter.compute_weights(X_resampled[adjustment_set], X_resampled[[TREATMENT]], bin_edges_contained)

            iptw_params = {
                "n_estimators": 200,
                "depth": None,
                "min_data_in_leaf": round(X_resampled.shape[0]*0.01),
                "learning_rate": 0.01,
                "subsample": 1,
                "rsm": 1,
                "objective": "Logloss",
                "silent": True,
                "l2_leaf_reg": 3
            }
            iptw_params["monotone_constraints"] = monotone_constrains
            iptw_model = CatBoostClassifier(**iptw_params)
            iptw_boostrap = iptw_estimation(X_resampled, y_resampled, weights_iptw, features_model, iptw_model)
            individual_potential_outcome = individual_dose_response_curve(
                df_eval=intervention_df,
                treatment_interventions=intervention_values,
                predictive_model=iptw_boostrap, 
                modelling_features=features_model, 
                feature_counterfactual=TREATMENT, 
                model_package="sklearn",
                task="classification"
            )

        # get results
        individual_real_outcome = individual_real_dose_response_curve(
            df_eval=intervention_df, 
            treatment_interventions=intervention_values,
            generator=generator,
            feature_counterfactual=TREATMENT,
        )
        individual_potential_outcomes_resample = np.array(individual_potential_outcome)

        individual_real_outcomes_resample = np.array(individual_real_outcome)

        average_dose_response_curve_resample = np.mean(individual_potential_outcomes_resample, axis=0)
        error_resample = compute_arrays_estimation_rmse(individual_potential_outcomes_resample, individual_real_outcomes_resample)
        
        average_dose_response_curves.append(average_dose_response_curve_resample)
        errors_estimation.append(error_resample)

    # Convert predictions to a NumPy array
    average_dose_response_curves = np.array(average_dose_response_curves)
    errors_estimation = np.array(errors_estimation)

    print(average_dose_response_curves)
    print(errors_estimation)

    # Average
    average_dose_response_curve = np.mean(average_dose_response_curves, axis=0)
    average_estimation_error = np.mean(errors_estimation, axis=0)


    # Confidence intervals
    lower_bound_dose_response = np.percentile(average_dose_response_curves, (1 - confidence_level) * 100, axis=0)
    upper_bound_dose_response = np.percentile(average_dose_response_curves, confidence_level * 100, axis=0)

    lower_bound_estimation_error = np.percentile(errors_estimation, (1 - confidence_level) * 100, axis=0)
    upper_bound_estimation_error = np.percentile(errors_estimation, confidence_level * 100, axis=0)

    results = {
        "dose-response": {
            "average": average_dose_response_curve,
            "lower_bound": lower_bound_dose_response,
            "upper_bound": upper_bound_dose_response
        },
        "estimation-error": {
            "average": average_estimation_error,
            "lower_bound": lower_bound_estimation_error,
            "upper_bound": upper_bound_estimation_error
        },

    }
    return results