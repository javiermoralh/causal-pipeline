import numpy as np
import statsmodels.formula.api as smf

from sklearn.base import clone
from catboost import CatBoostClassifier, CatBoostRegressor

from config import TREATMENT, OUTCOME
from utils.linear_regression import build_sm_regression_formula



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
    ).fit()
    return logit_estimator

def s_learner_estimation(X_train_obs, y_train_obs, features, model):
    s_learner_estimator = clone(model)
    s_learner_estimator.fit(
        X_train_obs[features].copy().to_numpy(),
        y_train_obs.to_numpy(),
    )
    return s_learner_estimator


def iptw_estimation(X_train_obs, y_train_obs, weight, features, model):
    iptw_estimator = clone(model)
    iptw_estimator.fit(
        X_train_obs[features].copy().to_numpy(),
        y_train_obs.to_numpy(),
        sample_weight=weight
    )
    return iptw_estimator
