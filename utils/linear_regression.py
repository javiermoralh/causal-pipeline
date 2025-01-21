import pandas as pd


def add_interactions(df, interaction_string):
    """
    Add interaction terms to dataframe based on string specification
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    interaction_string : str
        String specifying interactions, format: "var1*var2;var3*var4"
        
    Returns:
    --------
    tuple: (pandas.DataFrame, list)
        - DataFrame with added interaction terms
        - List of interaction term names
    """
    # Create copy of dataframe
    df_with_interactions = df.copy()
    
    # List to store interaction names
    interaction_names = []
    
    # Parse interactions from string
    interaction_pairs = interaction_string.split(';')
    
    for pair in interaction_pairs:
        vars_to_interact = pair.split('*')
        if len(vars_to_interact) != 2:
            raise ValueError(f"Invalid interaction specification: {pair}")
            
        var1, var2 = vars_to_interact[0].strip(), vars_to_interact[1].strip()
        if var1 not in df.columns or var2 not in df.columns:
            raise ValueError(f"Variables not found: {var1} or {var2}")
            
        interaction_name = f"{var1}_{var2}_interaction"
        df_with_interactions[interaction_name] = df[var1] * df[var2]
        interaction_names.append(interaction_name)
    
    return df_with_interactions, interaction_names


def discretize_features(
    df: pd.DataFrame, 
    bin_dict: dict, 
    add_dummies: bool = True, 
    drop_first: bool = True,
    right: bool = True,
    drop_unnecesary_features: bool = True
) -> pd.DataFrame:
    """
    Discretize (bin) numeric columns in `df` based on the provided bin edges.
    
    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame.
    bin_dict : dict
        A dictionary where keys are column names in `df` and
        values are lists (or arrays) of bin edges for that column.
        Example:
          {
            "n_cuenta": [-0.001, 1, 3, 5, 10],
            "n_tarjeta": [0, 1, 3, 5]
          }
    add_dummies : bool, default True
        Whether to create one-hot encoded dummy columns for each binned feature.
    drop_first : bool, default True
        If `add_dummies` is True, whether to drop the first dummy column 
        (making it the reference category).
    right : bool, default True
        Indicates whether the bins include the rightmost edge or not.
    
    Returns
    -------
    pd.DataFrame
        A copy of the original DataFrame with additional columns:
        - "<col>_bin" for each binned column
        - If add_dummies=True, dummy columns for each bin.
    """
    # Make a copy so we don't modify the original DataFrame in-place
    df_out = df.copy()

    for col, bins in bin_dict.items():
        # 1) Create a binned (categorical) column for this feature
        binned_col_name = f"{col}_bin"
        
        df_out[binned_col_name] = pd.cut(
            df_out[col],
            bins=bins,
            right=right,
            include_lowest=True
        )
        
        # 2) Optionally create dummy columns
        if add_dummies:
            dummies = pd.get_dummies(df_out[binned_col_name], prefix=binned_col_name, drop_first=drop_first)
            df_out = pd.concat([df_out, dummies], axis=1)
            
        if drop_unnecesary_features:
            df_out = df_out.drop(columns=[col, binned_col_name])
            
    return df_out


def build_sm_regression_formula(
    outcome: str,
    treatment: str,
    confounders: list[str],
    interactive_features: list[str],
    treatment_as_target: bool = False
) -> str:
    """
    Build a Statsmodels-compatible formula string. Can generate two types of formulas:
    1) Standard case (treatment_as_target=False):
       outcome ~ treatment + confounders + interactions
    2) Treatment as target (treatment_as_target=True):
       treatment ~ confounders + interactions with outcome

    Parameters
    ----------
    outcome : str or None
        The name of the outcome variable. Can be None if treatment_as_target=True.
    treatment : str
        The name of the treatment variable.
    confounders : list of str
        Variable names that enter additively.
    interactive_features : list of str
        Variable names that will each be interacted with treatment (or outcome if
        treatment_as_target=True).
    treatment_as_target : bool, default=False
        If True, generates formula with treatment as the dependent variable.

    Returns
    -------
    formula : str
        A string suitable for passing to statsmodels.formula.api.ols.
    """
    def safe_varname(varname: str) -> str:
        special_chars = set("()[],:+*-/ ")
        if any(char in varname for char in special_chars):
            return f"Q('{varname}')"
        else:
            return varname

    terms = []
    
    if treatment_as_target:
            
        # Add confounders
        terms += [safe_varname(c) for c in confounders]
        
        # Add interactive features and their interactions with outcome
        for feat in interactive_features:
            feat_safe = safe_varname(feat)
            terms.append(feat_safe)  # just the feature itself if no outcome
                
        # Build formula with treatment as target
        formula = f"{safe_varname(treatment)} ~ " + " + ".join(terms)
        
    else:
        # Original behavior when treatment is a predictor
        if outcome is None:
            raise ValueError("outcome must be provided when treatment_as_target=False")
            
        terms = [safe_varname(treatment)]  # treatment first
        terms += [safe_varname(c) for c in confounders]  # confounders
        
        # Interactive features and their interactions with treatment
        for feat in interactive_features:
            feat_safe = safe_varname(feat)
            treat_safe = safe_varname(treatment)
            terms.append(f"{feat_safe}:{treat_safe}")
            
        formula = f"{safe_varname(outcome)} ~ " + " + ".join(terms)

    return formula