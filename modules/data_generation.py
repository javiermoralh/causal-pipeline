import pandas as pd
import numpy as np

from scipy import stats

from config import TREATMENT, OUTCOME, SEED
from sklearn.preprocessing import StandardScaler

np.random.seed(SEED)

class DataGeneration:
    def __init__(self):
        # Initialize scalers for different feature groups
        self.confounder_scaler = StandardScaler()
        self.treatment_only_scaler = StandardScaler()
        self.outcome_only_scaler = StandardScaler()
        
        # Define feature groups
        self.confounders = [
            'years_since_default',
            'default_debt_amount',
            'n_loans',
            'debt_cirbe',
            'n_cards'
        ]
        self.treatment_only = ['loss_given_default', 'n_refin']
        self.outcome_only = [
            'years_history',
            'n_accounts',
            'months_since_first_payment'
        ]
        self.fitted = False

    def generate_random_financial_data(self, n_samples):
        """Generate random financial data just like the original function"""
        data = {}

        # Start with credit history (antiguedad) as it will influence other variables
        data['years_history'] = stats.gamma.rvs(a=4, scale=5, size=n_samples)

        # Number of accounts slightly influenced by credit history
        base_accounts = stats.poisson.rvs(mu=2, size=n_samples)
        credit_factor = np.clip(data['years_history'] / 10, 0, 2)  # More credit history = more accounts
        data['n_accounts'] = np.random.poisson(base_accounts * credit_factor*0.1)

        # Number of loans
        base_loans = stats.poisson.rvs(mu=1.5, size=n_samples)
        data['n_loans'] = base_loans 

        # Years in default
        data['years_since_default'] = stats.gamma.rvs(a=2, scale=2, size=n_samples)

        # LGD
        data['loss_given_default'] = stats.beta.rvs(11, 5, size=n_samples)

        # Default debt amount slightly correlated with number of loans and LGD
        loan_factor = np.clip(data['n_loans'] / 2, 0, 3)  # More loans = higher debt
        debt_base = stats.lognorm.rvs(s=1, scale=np.exp(10), size=n_samples)
        data['default_debt_amount'] = debt_base * (1 + 0.1 * loan_factor)

        # Number of credit cards slightly correlated with credit history
        zeros = np.random.binomial(1, 0.1 / (1 + 0.05 * credit_factor), n_samples)
        base_cards = stats.poisson.rvs(mu=2.5, size=n_samples)
        data['n_cards'] = base_cards * (1 - zeros)

        # Number of refinances
        base_refinances = stats.poisson.rvs(mu=1, size=n_samples)
        data['n_refin'] = base_refinances

        # Months since first payment correlated with years in default
        base_months = stats.expon.rvs(scale=40, size=n_samples)
        data['months_since_first_payment'] = base_months

        # External debt
        external_debt_base = stats.lognorm.rvs(s=0.8, scale=np.exp(9), size=n_samples)
        data['debt_cirbe'] = external_debt_base


        return pd.DataFrame(data)

    def fit_scalers(self, df):
        """Fit scalers on the initial dataset"""
        # Log transform monetary values before scaling
        df_transformed = df.copy()
        df_transformed['default_debt_amount'] = np.log1p(df['default_debt_amount'])
        df_transformed['debt_cirbe'] = np.log1p(df['debt_cirbe'])
        
        # Fit scalers on respective feature groups
        self.confounder_scaler.fit(df_transformed[self.confounders])
        self.treatment_only_scaler.fit(df_transformed[self.treatment_only])
        self.outcome_only_scaler.fit(df_transformed[self.outcome_only])
        
        self.fitted = True

    def transform_features(self, df, noise=False, noise_scale=0.2):
        """Transform features using fitted scalers and optionally add noise
        
        Args:
            df: Input DataFrame
            noise: Whether to add Gaussian noise after standardization
            noise_scale: Standard deviation of the Gaussian noise
        """
        if not self.fitted:
            raise ValueError("Scalers must be fitted before transforming!")
            
        df_transformed = df.copy()
        
        # Log transform monetary values
        df_transformed['default_debt_amount'] = np.log1p(df['default_debt_amount'])
        df_transformed['debt_cirbe'] = np.log1p(df['debt_cirbe'])
        
        # Transform each feature group and add noise if requested
        def transform_with_noise(scaler, features):
            transformed = scaler.transform(df_transformed[features])
            if noise:
                transformed += np.random.normal(0, noise_scale, size=transformed.shape)
            return transformed
        
        df_transformed[self.confounders] = transform_with_noise(self.confounder_scaler, self.confounders)
        df_transformed[self.treatment_only] = transform_with_noise(self.treatment_only_scaler, self.treatment_only)
        df_transformed[self.outcome_only] = transform_with_noise(self.outcome_only_scaler, self.outcome_only)
        
        return df_transformed

    def generate_treatment(self, df, noise=True,  noise_scale=0.2, treament_noise_std=5):
        """Generate treatment using scaled features"""
        df_scaled = self.transform_features(df, noise=noise, noise_scale=noise_scale)
        
        # Linear combination of scaled features
        linear_part = (
            0.5 * df_scaled['years_since_default'] +
            0.4 * df_scaled['default_debt_amount'] +
            0.3 * df_scaled['n_loans'] +
            0.3 * df_scaled['debt_cirbe'] +
            0.2 * df_scaled['n_cards'] +
            0.3 * df_scaled['loss_given_default'] +
            0.2 * df_scaled['n_refin']
        )
        
        # Interactions and non-linear terms
        linear_part += (
            0.1 * df_scaled['years_since_default'] * df_scaled['default_debt_amount'] +
            0.1 * np.square(df_scaled['n_loans'])
        )
        
        # Convert to [0,100] range
        propensity = 1 / (1 + np.exp(-linear_part))
        treatment_raw = 100 * propensity
        
        # Add noise if requested
        if noise:
            # treatment_noise = np.random.normal(loc=0, scale=treament_noise_std, size=len(df))
            # treatment = treatment_raw + treatment_noise

            # draw each treatment from a truncated normal distribution so that it stays in [0, 100]
            scale = treament_noise_std
            treatment = np.zeros_like(treatment_raw)
            for i, base_value in enumerate(treatment_raw):
                # Lower/upper bounds in "Z-space"
                a = (0    - base_value) / scale
                b = (100  - base_value) / scale
                # Draw from truncated normal
                treatment[i] = stats.truncnorm.rvs(a, b, loc=base_value, scale=scale)

        else:
            treatment = treatment_raw
            
        return np.clip(treatment, 0, 100)

    def calculate_outcome_probability(self, df, treatment):
        """Calculate outcome probability using scaled features"""
        df_scaled = self.transform_features(df)
        
        # Normalize treatment to [0,1]
        t_norm = treatment / 100.0
        
        # Calculate log exponent using scaled features
        log_e = (
            0.6 * df_scaled['years_since_default'] +
            0.5 * df_scaled['default_debt_amount'] +
            0.5 * df_scaled['n_loans'] +
            0.4 * df_scaled['debt_cirbe'] +
            0.3 * df_scaled['n_cards']
        )
        
        # Subtract outcome-only features
        log_e -= (
            0.4 * df_scaled['years_history'] +
            0.3 * df_scaled['n_accounts'] +
            0.2 * df_scaled['months_since_first_payment']
        )
        
        # Add interactions
        log_e += (
            0.1 * df_scaled['years_since_default'] * df_scaled['default_debt_amount'] +
            0.1 * np.square(df_scaled['n_loans'])
        )
        
        # Calculate probability
        e_i = np.exp(log_e)
        prob = t_norm ** e_i
        
        # Force boundary conditions
        prob = np.where(treatment == 0, 0, prob)
        prob = np.where(treatment == 100, 1, prob)

        probs = np.clip(prob, 0, 1)
        outcome = np.random.binomial(n=1, p=probs)
        
        return probs, outcome

    def compute_causal_effects(self, df, treatment_values, aggregation="ate"):
        """Compute causal effects using the scaled features"""
        effects = []
        
        original_treatments = df[TREATMENT].to_numpy().flatten()
        original_outcomes = df[OUTCOME].to_numpy().flatten()
        for value_treatment in treatment_values:
            df_interventions = df.copy()
            df_interventions[TREATMENT] = value_treatment
            probs, _ = self.calculate_outcome_probability(
                df_interventions, 
                df_interventions[TREATMENT]
            )
            outcomes = []
            for prob, outcome, original_t in zip(probs, original_outcomes, original_treatments):
                if (outcome == 1) & (value_treatment >= original_t):
                    outcomes.append(1)
                elif (outcome == 0) & (value_treatment <= original_t):
                    outcomes.append(0)
                else:
                    outcomes.append(prob)
            
            outcomes = np.array(outcomes)
            if aggregation == "ate":
                effects.append(outcomes.mean())
            elif aggregation == "ite":
                effects.append(outcomes.tolist())
                
        if aggregation == "ite":
            effects = [list(row) for row in zip(*effects)]
            
        return effects


def add_synthetic_features(df, n_redundant=3, n_noise=3, random_state=SEED):
    """
    Add redundant and noisy features to a financial dataset.
    Redundant features are created with target correlations
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original dataframe with financial features
    n_redundant : int
        Number of redundant features to generate
    n_noise : int
        Number of noisy features to generate
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional synthetic features
    """
    result_df = df.copy()
    
    # List of original features to sample from
    original_features = [c for c in df.columns.tolist() if c not in [TREATMENT, OUTCOME]]
    
    # Generate redundant features
    for i in range(n_redundant):
        # Randomly select a base feature
        base_feature = np.random.choice(original_features)
        base_data = df[base_feature].values
        
        # Standardize the base feature
        base_data = (base_data - base_data.mean()) / base_data.std()
        
        # Generate target correlation (randomly positive or negative)
        sign = np.random.choice([-1, 1])
        target_corr = sign * np.random.uniform(0.1, 0.9)
        
        # Generate correlated data using Cholesky decomposition
        # Create 2x2 correlation matrix
        corr_matrix = np.array([[1.0, target_corr],
                              [target_corr, 1.0]])
        
        # Compute Cholesky decomposition
        L = np.linalg.cholesky(corr_matrix)
        
        # Generate independent standard normal data
        uncorrelated = np.random.standard_normal(len(df))
        
        # Create correlated data
        data = np.dot(L, np.vstack((base_data, uncorrelated)))
        redundant_feature = data[1]  # Take the second row
        
        # Add some additional noise to make it more realistic
        noise_factor = np.random.uniform(0.1, 0.2)  # Reduced noise factor
        noise = np.random.normal(0, noise_factor, len(df))
        redundant_feature += noise
        
        # Verify correlation
        actual_corr = np.corrcoef(base_data, redundant_feature)[0, 1]
        
        # Name and add the feature
        result_df[f'redundantfeature{i+1}_{base_feature}'] = redundant_feature
        
    # Generate noisy features from various financial distributions
    for i in range(n_noise):
        distribution = np.random.choice(['lognormal', 'gamma', 'beta', 'pareto'])
        scalar = np.random.randint(2, 5) * np.random.choice([-1, 1])
        
        if distribution == 'lognormal':
            # Typical for asset returns, prices
            mu, sigma = np.random.uniform(0, 2), np.random.uniform(0.2, 0.8)
            noise = stats.lognorm.rvs(s=sigma, scale=np.exp(mu), size=len(df))
        elif distribution == 'gamma':
            # Typical for waiting times, loan durations
            shape = np.random.uniform(1, 5)
            scale = np.random.uniform(0.5, 2)
            noise = stats.gamma.rvs(a=shape, scale=scale, size=len(df))
        elif distribution == 'beta':
            # Typical for rates, proportions
            a = np.random.uniform(1, 5)
            b = np.random.uniform(1, 5)
            noise = stats.beta.rvs(a, b, size=len(df))
        else:  # pareto
            # Typical for extreme events, large losses
            b = np.random.uniform(2, 5)  # shape parameter
            noise = stats.pareto.rvs(b, size=len(df))
            
        # Scale the noise to have reasonable magnitude
        noise = ((noise - noise.mean()) / noise.std()) * scalar
        result_df[f'noise*feature*{i+1}'] = noise
        
    return result_df

