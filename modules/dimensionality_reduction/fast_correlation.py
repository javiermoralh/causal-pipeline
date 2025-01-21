from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class FCBFSelector(BaseEstimator, TransformerMixin):
    """

    A feature selector that implements the Fast Correlation-Based Filter (FCBF) algorithm.

    

    The selector follows these steps:



    Feature Selection Process:

       a) Calculates Symmetrical Uncertainty (SU) between each feature and target (relevance)

       b) Removes features with SU below threshold

       c) Among remaining features, removes redundant ones based on SU comparison

       d) A feature Fj is considered redundant to Fi if SU(Fi,Fj) >= SU(Fj,class)

    

    Parameters:

    -----------

    threshold : float, default=0.01

        Threshold for selecting relevant features based on Symmetrical Uncertainty (SU).

        SU ranges from 0 (features are completely independent) to 1 (perfect correlation).

        

        The threshold of 0.01 is recommended:

        - SU < 0.01: Usually indicates weak or noisy relationships

        - SU ≥ 0.01: Indicates meaningful feature-target dependencies

        

        This threshold value provides a good balance for diverse datasets by:

        - Removing noisy or irrelevant features (SU < 0.01)

        - Keeping moderately to strongly relevant features

        - Being robust across different dataset sizes and characteristics

        - Maintaining computational efficiency

    """

    

    def __init__(self, threshold=0.01):
        self.threshold = threshold
        self.selected_features_ = None
        self.relevance_scores_ = None

    def fit(self, X, y):
        """
        Fit the FCBF selector to the data.

        

        Parameters:

        -----------

        X : pd.DataFrame

            Input features

        y : pd.Series or array-like

            Target variable

        

        Returns:

        --------

        self : object

            Returns the instance itself
        """     

        # Calculate SU scores between features and target
        self.relevance_scores_ = self._calculate_su_scores(X, y)

        # Select features using FCBF
        self.selected_features_ = self._select_features_fcbf(X)

        return self

    def transform(self, X):
        """
        Transform the data by selecting only the chosen features.

        Parameters:

        -----------

        X : pd.DataFrame

            Input features

            

        Returns:

        --------

        pd.DataFrame

            Transformed data with only selected features
        """
        if self.selected_features_ is None:
            raise ValueError("Transformer has not been fitted yet.")

        return X[self.selected_features_]

    
    def _calculate_entropy(self, x):
        """

        Calculate Shannon entropy for a variable.

        

        Parameters:

        -----------

        x : pd.Series

            Input variable  

        Returns:

        --------

        float : Entropy value

        """

        _, counts = np.unique(x, return_counts=True)

        probabilities = counts / len(x)

        return -np.sum(probabilities * np.log2(probabilities))

    

    def _calculate_conditional_entropy(self, x, y):

        """

        Calculate conditional entropy H(X|Y).

        

        H(X|Y) = -∑(y)P(y)∑(x)P(x|y)log₂P(x|y)

        

        Parameters:

        -----------

        x : pd.Series

            First variable

        y : pd.Series

            Second variable

            

        Returns:

        --------

        float : Conditional entropy value

        """

        # Create joint frequency table

        joint_counts = pd.crosstab(y, x)

        total_samples = joint_counts.sum().sum()

        

        # Calculate conditional entropy

        conditional_entropy = 0

        

        # For each value of Y

        for y_val in joint_counts.index:

            # P(Y = y)

            py = joint_counts.loc[y_val].sum() / total_samples

            

            if py > 0:

                # Get conditional probabilities P(X|Y=y)

                x_counts_given_y = joint_counts.loc[y_val]

                px_given_y = x_counts_given_y / x_counts_given_y.sum()

                

                # Calculate -P(y)∑P(x|y)log₂P(x|y) for this y value

                # Only include non-zero probabilities in log calculation

                nonzero_probs = px_given_y[px_given_y > 0]

                if len(nonzero_probs) > 0:

                    conditional_entropy -= py * np.sum(

                        nonzero_probs * np.log2(nonzero_probs)

                    )

        

        return conditional_entropy

    

    def _calculate_symmetrical_uncertainty(self, x, y):

        """

        Calculate Symmetrical Uncertainty (SU) between two variables.

        SU(X,Y) = 2 * [H(X) - H(X|Y)] / [H(X) + H(Y)]

        

        Parameters:

        -----------

        x : pd.Series

            First variable

        y : pd.Series

            Second variable

            

        Returns:

        --------

        float : SU value between 0 and 1

        """

        h_x = self._calculate_entropy(x)

        h_y = self._calculate_entropy(y)

        

        if h_x == 0 or h_y == 0:

            return 0

        

        h_x_given_y = self._calculate_conditional_entropy(x, y)

        information_gain = h_x - h_x_given_y

        

        return 2 * information_gain / (h_x + h_y)

    

    def _calculate_su_scores(self, X, y):

        """

        Calculate SU scores between each feature and the target.

        

        Parameters:

        -----------

        X : pd.DataFrame

            Input features

        y : pd.Series

            Target variable

            

        Returns:

        --------

        dict : Feature names to SU scores

        """

        su_scores = {}

        for feature in X.columns:

            su = self._calculate_symmetrical_uncertainty(X[feature], y)

            su_scores[feature] = su

            

        return su_scores

    

    def _select_features_fcbf(self, X):

        """

        Select features using the FCBF algorithm.

        

        1. Sort features by SU with target

        2. Remove features below threshold

        3. For remaining features, remove redundant ones

        

        Parameters:

        -----------

        X : pd.DataFrame

            Input features

            

        Returns:

        --------

        list : Selected feature names

        """

         # Sort features by SU value

        sorted_features = sorted(

            self.relevance_scores_.items(),

            key=lambda x: x[1],

            reverse=True

        )

        

        # Filter features above threshold

        relevant_features = [

            feature for feature, su in sorted_features 

            if su > self.threshold

        ]

        

        # Remove redundant features

        final_features = []

        remaining_features = relevant_features.copy()  # Create a copy to iterate

        

        while remaining_features:

            # Get feature with highest SU

            fi = remaining_features[0]

            final_features.append(fi)

            

            # Remove the current feature from remaining features

            remaining_features.remove(fi)

            

            # Create a copy of remaining features to modify during iteration

            features_to_remove = []

            

            # Compare with remaining features

            for fj in remaining_features:

                su_ij = self._calculate_symmetrical_uncertainty(X[fi], X[fj])

                if su_ij >= self.relevance_scores_[fj]:

                    features_to_remove.append(fj)

            

            # Remove redundant features

            for feature in features_to_remove:

                if feature in remaining_features:

                    remaining_features.remove(feature)

        

        return final_features

    

    def get_feature_scores(self):

        """

        Get the SU scores for selected features.

        

        Returns:

        --------

        dict

            Dictionary mapping selected features to their SU scores

        """

        if self.selected_features_ is None:

            raise ValueError("Transformer has not been fitted yet.")

            

        return {

            feature: self.relevance_scores_[feature] 

            for feature in self.selected_features_

        }