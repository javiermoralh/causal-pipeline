from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from typing import Union, List, Optional
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator

from modules.dimensionality_reduction.fast_correlation import FCBFSelector
from modules.dimensionality_reduction.sequential_forward_selector import SFSCV
from modules.dimensionality_reduction.funcs import  (
    encode_categorical_features, kmeans_discretize, ImputeMissing
)
from config import TREATMENT


class HybridSelector(BaseEstimator, TransformerMixin):

    """

    Hybrid feature selection combining FCBF and Sequential Forward Floating Selection.

    

    Process:

    1. FCBF filter phase: Removes clearly irrelevant features using Symmetrical Uncertainty. 

    The idea is to use a less restrictive threshold than in standard FCBF to select more features.

    2. SFFS wrapper phase: Fine-tunes selection using cross-validated model performance

    

    This hybrid approach balances computational efficiency with selection accuracy by:

    - Using FCBF to quickly eliminate obviously irrelevant features

    - Applying SFFS on the reduced feature set for precise selection

    

    Parameters:

        model: BaseEstimator

            Sklearn-compatible model for SFFS phase

        scorer: callable

            Scoring function for SFFS phase

        metric_direction: str, default='maximize'

            'maximize' for metrics like accuracy, 'minimize' for errors

        cv: BaseCrossValidator, optional

            Cross-validation strategy

        categorical_features: List[str], optional

            Categorical feature names for model

        min_improvement: float, default=1e-4

            Minimum improvement threshold for SFFS

        task: str, default='regression'

            Type of ML task: 'regression', 'binary_classification', 'multiclass_classification'

        fcbf_threshold: float, default=0.001

            FCBF score threshold for initial filtering



    """

    

    def __init__(self,

                 model: BaseEstimator,

                 scorer: callable,

                 metric_direction: str = 'minimize',

                 cv: Optional[BaseCrossValidator] = None,

                 categorical_features: Optional[List[str]] = None,

                 min_improvement: float = 1e-4,

                 task: str = 'regression',

                 fcbf_threshold: float = 0.005,
                 include_treatment: bool = False) -> None:

        

        self.model = model

        self.scorer = scorer

        self.metric_direction = metric_direction

        self.cv = cv

        self.categorical_features = categorical_features

        self.min_improvement = min_improvement

        self.task = task

        self.fcbf_threshold = fcbf_threshold
        self.include_treatment = include_treatment

        

        # Initialize individual selectors

        self.fcbf = FCBFSelector(threshold=self.fcbf_threshold)

        self.sfs = SFSCV(

            model=self.model,

            scorer=self.scorer,

            metric_direction=self.metric_direction,

            cv=self.cv,

            categorical_features=self.categorical_features,

            min_improvement=self.min_improvement,

            task=self.task,
            include_treatment=self.include_treatment,

        )

        

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]) -> 'HybridFcbfSfs':

        """

        Fit the hybrid selector using two-phase selection.

        

        Parameters:

            X: Feature matrix

            y: Target variable

            

        Returns:

            self: Fitted selector

        """

                

        # Phase 1: FCBF filtering

        # First fit FCBF to get selected features

        

        X_processed, y_processed = self.process_filter(X, y)

        self.fcbf.fit(X_processed, y_processed)

        if self.include_treatment:
            self.fcbf.selected_features_.append(TREATMENT)

        selected_columns = self.fcbf.selected_features_
        print(f"Selected columns by FCBF: {len(selected_columns)}")
        print(selected_columns)

        

        if len(selected_columns) == 0:

            # If no features are selected by NMI, use all features

            selected_columns = X.columns.tolist()

        

        # Phase 2: Run SFS on original dataset with only FCBF-selected columns

        X_filtered = X[selected_columns]

        

        # Update categorical features list for selected features only

        if self.categorical_features:

            self.sfs.categorical_features = [

                f for f in self.categorical_features 

                if f in selected_columns

            ]

        

        

        self.sfs.fit(X_filtered, y)

        

        # Store the final selected features

        self.selected_features_ = self.sfs.selected_features_

        

        return self

    

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        """

        Transform data using selected features.

        

        Parameters:

            X: Feature matrix

            

        Returns:

            pd.DataFrame: Transformed data with selected features

        """

        return X[self.selected_features_]



    

    def fit_transform(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:

        """

        Fit and transform in one step.

        

        Parameters:

            X: Feature matrix

            y: Target variable

            

        Returns:

            pd.DataFrame: Transformed data with selected features

        """

        return self.fit(X, y).transform(X)

    

    def process_filter(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]):

        """

        Process data for the filter phase.

        """

        X_transformed = X.copy()

        y_transformed = y.copy()

        

        # Handle target variable - ensure it's a Series

        if isinstance(y_transformed, pd.DataFrame):

            y_transformed = y_transformed.iloc[:, 0]  # Take first column if DataFrame

        else:

            y_transformed = pd.Series(y_transformed)

            

        # Impute if there are missing values

        if X_transformed.isna().any().any():

            X_transformed = ImputeMissing().fit_transform(X_transformed)

        

        # Encode categorical variables

        X_transformed = encode_categorical_features(X_transformed)

        

        # Discretize numeric variables

        X_transformed = kmeans_discretize(X_transformed, num_bins=5)

        

        if self.task == 'regression':

            y_transformed = kmeans_discretize(y_transformed, num_bins=5)

            

        return X_transformed, y_transformed