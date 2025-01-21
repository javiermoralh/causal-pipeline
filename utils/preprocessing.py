import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer
from config import TREATMENT, OUTCOME


def filter_treatment_ranges(df):
    df = df[
        (df[TREATMENT] >= 5)
        & (df[TREATMENT] <= 80)
    ]
    return df

def create_stratification_split_col(df, treatment_col, outcome_col):
    strat_data = df.copy()
    kbd = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    treatment_binned = kbd.fit_transform(strat_data[[treatment_col]])
    treatment_binned = treatment_binned.ravel()
    strat_data['strat'] = (
        pd.Series(treatment_binned, index=strat_data.index).astype(str) 
        + '_' + strat_data[outcome_col].astype(str)
    )
    return strat_data['strat']

    