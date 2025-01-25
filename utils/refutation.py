import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from config import TREATMENT, OUTCOME
from sklearn.base import clone
from catboost import CatBoostClassifier, CatBoostRegressor

from utils.potential_outcomes import (
    individual_dose_response_curve,
    average_dose_response_curve,
)
from utils.estimates import logistic_regression_estimation, s_learner_estimation, augmented_iptw_estimation
from modules.iptw import ContinuousIPTW



def random_var(
    data_var=None, distribution="permutation", sample_size=100, random_state=42
):
    np.random.seed(random_state)
    if distribution == "permutation":
        var_random = np.random.permutation(data_var)
    if (distribution == "uniform") and (data_var is not None):
        var_random = np.random.random_integers(
            low=data_var.min(), high=data_var.max(), size=sample_size
        )
    if (distribution == "normal") and (data_var is not None):
        var_random = np.random.normal(
            loc=data_var.mean(), scale=data_var.std(), size=sample_size
        )
    if (distribution == "normal") and (data_var is None):
        var_random = np.random.normal(loc=0.0, scale=1.0, size=sample_size)
    return var_random


def evalue_for_rr(rr):
    """
    Computes the VanderWeele-style E-value for a risk ratio (RR) or odds ratio (>=1).
    If rr < 1, we invert it first to ensure the formula is used consistently.
    """
    # If rr < 1, invert it.
    rr_adj = rr if rr >= 1 else 1.0 / rr
    
    # If rr_adj == 1, the E-value is 1 (no effect).
    if rr_adj == 1:
        return 1.0
    
    return rr_adj + np.sqrt(rr_adj * (rr_adj - 1.0))


def dose_response_evalue_continuous(
    treatment_values,
    dose_response_curve,
    reference_dose_idx=0,
    random_state=0
):
    """
    Compute a single 'global' E-value that would, under naive assumptions, 
    be sufficient to 'explain away' the entire observed dose-response pattern.

    Notes
    -----
    - This function uses a naive ratio of means as 'the effect measure'.
    - It then applies a bounding argument that a single confounder would have to be
      large enough to explain the maximum ratio departure from the reference across *all* doses.
    - Real analyses may require more sophisticated parametric modeling or assumptions.
    """
    np.random.seed(random_state)
    
    # 1) If needed, define reference_dose
    if reference_dose_idx is None:
        reference_dose_idx = 0
    
    
    # 2) Identify the reference group (the bin that contains reference_dose)
    #    We'll pick the group that is numerically closest to reference_dose.
    reference_outcome = dose_response_curve[reference_dose_idx]
    
    # 3) Compute ratio to reference
    ratios = [c / reference_outcome  for c in dose_response_curve[reference_dose_idx+1:]]
    
    # 4) Compute E-value for each ratio, then define "global" E-value as the max
    def compute_pointwise_evalue(risk_ratio):
        return evalue_for_rr(risk_ratio)
    
    pointwise_e_values = [compute_pointwise_evalue(ratio) for ratio in ratios]
    global_evalue_est = np.max(pointwise_e_values)
    return pointwise_e_values


def get_ci_e_values(
    estimation_method,
    train_df,
    n_iterations,
    intervention_df,
    intervention_values,
    adjustment_set,
    outcome_causes,
    confidence_level=0.95
):

    average_dose_response_curves = []
    e_values_curves = []
    for i in range(n_iterations):
        # Create a bootstrapped dataset
        adjustment_set_iter = adjustment_set
        features_model = adjustment_set + [TREATMENT]
        train_df_resampled = train_df.sample(n=len(train_df), replace=True, random_state=i)
        X_train_iter, y_train_iter = (
            train_df_resampled[features_model],
            train_df_resampled[[OUTCOME]],
        )
        intervention_df_copy = intervention_df.copy()

        # train model and predict POs
        if estimation_method == "logistic_reg":
            if len(outcome_causes) > 0:
                binned_outcome_causes = [c  + '_bins' for c in outcome_causes]
                for outcome_cause in outcome_causes:
                    bins = pd.qcut(train_df[outcome_cause], q=5, retbins=True, duplicates='drop')[1]
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

        elif estimation_method == "s_learner":
            s_learner_params = {
                "n_estimators": 200,
                "depth": None,
                "min_data_in_leaf": round(X_train_iter.shape[0]*0.01),
                "learning_rate": 0.01,
                "subsample": 1,
                "rsm": 1,
                "objective": "Logloss",
                "silent": True,
                "l2_leaf_reg": 3
            }
            s_learner_model = CatBoostClassifier(**s_learner_params)
            s_learner_boostrap = s_learner_estimation(
                X_train_iter, y_train_iter, adjustment_set_iter+[TREATMENT], s_learner_model
            )
            individual_potential_outcome = individual_dose_response_curve(
                df_eval=intervention_df_copy,
                treatment_interventions=intervention_values,
                predictive_model=s_learner_boostrap, 
                modelling_features=adjustment_set_iter + [TREATMENT], 
                feature_counterfactual=TREATMENT, 
                model_package="sklearn",
                task="classification"
            )

        elif estimation_method == "augmented_iptw":
            init_params = {
                "n_estimators": 200,
                "depth": None,
                "min_data_in_leaf": round(X_train_iter.shape[0]*(2/3)*0.01),
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
            controls = [c for c in features_model if c != TREATMENT]
            weights_iptw = weighter.compute_weights(X_train_iter[controls], X_train_iter[[TREATMENT]], bin_edges_contained)

            iptw_params = {
                "n_estimators": 200,
                "depth": None,
                "min_data_in_leaf": round(X_train_iter.shape[0]*0.01),
                "learning_rate": 0.01,
                "subsample": 1,
                "rsm": 1,
                "objective": "Logloss",
                "silent": True,
                "l2_leaf_reg": 3
            }
            iptw_model = CatBoostClassifier(**iptw_params)
            iptw_boostrap = augmented_iptw_estimation(
                X_train_iter, y_train_iter, weights_iptw, features_model, iptw_model
                )
            individual_potential_outcome = individual_dose_response_curve(
                df_eval=intervention_df_copy,
                treatment_interventions=intervention_values,
                predictive_model=iptw_boostrap, 
                modelling_features=adjustment_set_iter + [TREATMENT], 
                feature_counterfactual=TREATMENT, 
                model_package="sklearn",
                task="classification"
            )

        # get results
        individual_potential_outcomes_resample = np.array(individual_potential_outcome)
        average_dose_response_curve_resample = np.mean(individual_potential_outcomes_resample, axis=0)
        e_values_curve = dose_response_evalue_continuous(
            intervention_values,
            average_dose_response_curve_resample,
            reference_dose_idx=0,
            random_state=0
        )

        average_dose_response_curves.append(average_dose_response_curve_resample)
        e_values_curves.append(e_values_curve)


    # Convert predictions to a NumPy array
    average_dose_response_curves = np.array(average_dose_response_curves)
    e_values_curves = np.array(e_values_curves)

    # Average
    average_dose_response_curve = np.mean(average_dose_response_curves, axis=0)
    average_e_values_curves = np.mean(e_values_curves, axis=0)


    # Confidence intervals
    lower_bound_e_values = np.percentile(e_values_curves, (1 - confidence_level) * 100, axis=0)
    upper_bound_e_values= np.percentile(e_values_curves, confidence_level * 100, axis=0)
    
    return average_e_values_curves, lower_bound_e_values, upper_bound_e_values


def get_ci_refutation_results(
    refutation,
    estimation_method,
    train_df,
    n_iterations,
    intervention_df,
    intervention_values,
    adjustment_set,
    outcome_causes,
    n_variables_common_cause=5,
    confidence_level=0.95
):

    average_dose_response_curves = []
    for i in range(n_iterations):
        # Create a bootstrapped dataset
        adjustment_set_iter = adjustment_set
        features_model = adjustment_set + [TREATMENT]
        train_df_resampled = train_df.sample(n=len(train_df), replace=True, random_state=i).copy()
        X_train_iter, y_train_iter = (
            train_df_resampled[features_model],
            train_df_resampled[[OUTCOME]],
        )
        intervention_df_copy = intervention_df.copy()
        if refutation == "placebo_treatment_replacement":
            random_treatment = random_var(
                X_train_iter[TREATMENT],
                distribution="permutation",
                sample_size=X_train_iter.shape[0],
                random_state=i,
            )
            X_train_iter[TREATMENT] = random_treatment
            train_df_resampled[TREATMENT] = random_treatment
        
        if refutation == "random_common_cause":
            common_causes = []
            for cc_i in range(1, n_variables_common_cause + 1):
                random_confounder = random_var(
                    None,
                    distribution="normal",
                    sample_size=X_train_iter.shape[0],
                    random_state=cc_i + i,
                )
                common_cause_name = "random_confounder_" + str(cc_i)
                X_train_iter[common_cause_name] = random_confounder
                train_df_resampled[common_cause_name] = random_confounder
                common_causes.append(common_cause_name)

            features_model = adjustment_set + [TREATMENT] + common_causes
            adjustment_set_iter = adjustment_set + common_causes
            for cc_i, common_cause_name in enumerate(common_causes):
                random_confounder_test = random_var(
                    None,
                    distribution="normal",
                    sample_size=intervention_df_copy.shape[0],
                    random_state=cc_i * i,
                )
                intervention_df_copy[common_cause_name] = random_confounder_test

        # train model and predict POs
        if estimation_method == "logistic_reg":
            if len(outcome_causes) > 0:
                binned_outcome_causes = [c  + '_bins' for c in outcome_causes]
                for outcome_cause in outcome_causes:
                    bins = pd.qcut(train_df[outcome_cause], q=5, retbins=True, duplicates='drop')[1]
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

        elif estimation_method == "s_learner":
            s_learner_params = {
                "n_estimators": 200,
                "depth": None,
                "min_data_in_leaf": round(X_train_iter.shape[0]*0.01),
                "learning_rate": 0.01,
                "subsample": 1,
                "rsm": 1,
                "objective": "Logloss",
                "silent": True,
                "l2_leaf_reg": 3
            }
            s_learner_model = CatBoostClassifier(**s_learner_params)
            s_learner_boostrap = s_learner_estimation(
                X_train_iter, y_train_iter, adjustment_set_iter+[TREATMENT], s_learner_model
            )
            individual_potential_outcome = individual_dose_response_curve(
                df_eval=intervention_df_copy,
                treatment_interventions=intervention_values,
                predictive_model=s_learner_boostrap, 
                modelling_features=adjustment_set_iter + [TREATMENT], 
                feature_counterfactual=TREATMENT, 
                model_package="sklearn",
                task="classification"
            )

        elif estimation_method == "augmented_iptw":
            init_params = {
                "n_estimators": 200,
                "depth": None,
                "min_data_in_leaf": round(X_train_iter.shape[0]*(2/3)*0.01),
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
            weights_iptw = weighter.compute_weights(
                X_train_iter[adjustment_set_iter], X_train_iter[[TREATMENT]], bin_edges_contained
            )

            iptw_params = {
                "n_estimators": 200,
                "depth": None,
                "min_data_in_leaf": round(X_train_iter.shape[0]*0.01),
                "learning_rate": 0.01,
                "subsample": 1,
                "rsm": 1,
                "objective": "Logloss",
                "silent": True,
                "l2_leaf_reg": 3
            }
            iptw_model = CatBoostClassifier(**iptw_params)
            iptw_boostrap = augmented_iptw_estimation(
                X_train_iter, y_train_iter, weights_iptw, adjustment_set_iter+[TREATMENT], iptw_model
            )
            individual_potential_outcome = individual_dose_response_curve(
                df_eval=intervention_df_copy,
                treatment_interventions=intervention_values,
                predictive_model=iptw_boostrap, 
                modelling_features=adjustment_set_iter + [TREATMENT], 
                feature_counterfactual=TREATMENT, 
                model_package="sklearn",
                task="classification"
            )

        # get results
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
    
    return average_dose_response_curve, lower_bound_dose_response, upper_bound_dose_response
