import numpy as np
import statsmodels.formula.api as smf

from sklearn.base import clone
from sklearn.model_selection import KFold

from config import TREATMENT, OUTCOME, SEED
from utils.linear_regression import build_sm_regression_formula
from modules.generealized_propensity_score import GPS

np.random.seed(SEED)

def logistic_regression_estimation(train_obs, confounders_list, outcome_causes=[]):
    regression_formula_str = build_sm_regression_formula(
        outcome=OUTCOME,
        treatment=TREATMENT,
        confounders=confounders_list,
        interactive_features=outcome_causes
    )
    logit_estimator = smf.logit(
        formula=regression_formula_str,
        data=train_obs
    ).fit(disp=False)
    return logit_estimator

def s_learner_estimation(X_train_obs, y_train_obs, features, model):
    s_learner_estimator = clone(model)
    s_learner_estimator.fit(
        X_train_obs[features].copy().to_numpy(),
        y_train_obs.to_numpy(),
    )
    return s_learner_estimator


def augmented_iptw_estimation(
        X_train_obs, 
        y_train_obs, 
        cv_folds,
        bin_edges,
        features,
        gps_controls, 
        propensity_model, 
        outcome_model, 
        final_model,
        model_calibration,
    ):

    # Propensity model (GPS)
    bin_edges_contained = bin_edges.copy()
    bin_edges_contained[0] = -1
    weighter = GPS(model=propensity_model, n_folds=cv_folds, random_state=SEED)
    weights_iptw = weighter.compute_weights(
        X_train_obs[gps_controls], X_train_obs[[TREATMENT]], bin_edges_contained, inverse_weights=True
    )
    p99 = np.percentile(weights_iptw, 99)
    weights_iptw = np.minimum(weights_iptw, p99)

    # Outcome model
    outcome_model_aiptw = clone(outcome_model)
    X_train_obs_copy = X_train_obs.copy()
    X_train_obs_copy["outcome_model_oos_predictions"] = 0.0

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    for train_idx, test_idx in kf.split(X_train_obs_copy):
        X_train_cv, X_val_cv = (
            X_train_obs_copy.iloc[train_idx, :], 
            X_train_obs_copy.iloc[test_idx, :]
        )
        y_train_cv = y_train_obs.iloc[train_idx, :][[OUTCOME]]
        outcome_model_aiptw_fold = clone(outcome_model_aiptw)
        outcome_model_aiptw_fold.fit(X_train_cv[features], y_train_cv)
        X_train_obs_copy.iloc[test_idx, -1] = outcome_model_aiptw_fold.predict_proba(X_val_cv)[:, 1]
    
    m_hat = X_train_obs_copy["outcome_model_oos_predictions"].to_numpy().flatten()
    pseudo_Y = m_hat + (y_train_obs.to_numpy().flatten() - m_hat) * weights_iptw


    # Final Model
    X_train_obs_copy["pseudo_Y"] = pseudo_Y
    X_train_obs_copy["final_model_oos_predictions"] = 0.0
    for train_idx, test_idx in kf.split(X_train_obs_copy):
        X_train_cv, X_val_cv = (
            X_train_obs_copy.iloc[train_idx, :], 
            X_train_obs_copy.iloc[test_idx, :]
        )
        y_train_cv = X_train_obs_copy.iloc[train_idx, :][["pseudo_Y"]]
        final_model_aiptw_fold = clone(final_model)
        final_model_aiptw_fold.fit(X_train_cv[features], y_train_cv)
        X_train_obs_copy.iloc[test_idx, -1] = final_model_aiptw_fold.predict(X_val_cv)

    # Refit
    final_model_iptw = clone(final_model)
    final_model_iptw.fit(
        X_train_obs[features].copy().to_numpy(),
        pseudo_Y
    )

    # Calibration
    final_model_aiptw_calib = clone(model_calibration)
    final_model_aiptw_calib.fit(
        X_train_obs_copy[["final_model_oos_predictions"]],
        y_train_obs
    )
    
    return final_model_iptw, final_model_aiptw_calib

def get_monotone_contrainst(features):
    monotone_constraints_list = []
    for f in features:
        if f != TREATMENT:
            monotone_constraints_list.append(0)
        else:
            monotone_constraints_list.append(1)
    return monotone_constraints_list
