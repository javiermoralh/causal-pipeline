import numpy as np
import pandas as pd
import random

from scipy.optimize import curve_fit
from config import TREATMENT, OUTCOME


class DomainPropagation:
    def __init__(
        self,
        outcome,
        treatment,
        perc_step_extrapolation,
        undersampling_perc_extrapolation=None,
        perc_step_interpolation=None,
        interpolation_method=None,
        undersampling_perc_interpolation=None,
        quadratic_power=None,
        treatment_range=[0, 100],
        stratified_subsampling=True,
        bin_edges=None,
        positivity_violations_dict=None,
        seed=42
    ) -> None:
        self.outcome = outcome
        self.treatment = treatment
        self.perc_step_extrapolation = perc_step_extrapolation
        self.perc_step_interpolation = perc_step_interpolation
        self.interpolation_method = interpolation_method
        self.undersampling_perc_extrapolation = undersampling_perc_extrapolation
        self.undersampling_perc_interpolation = undersampling_perc_interpolation
        self.interpolation_methods = [
            None,
            "linear_interpolation",
            "quadratic_interpolation",
        ]
        self.quadratic_power = quadratic_power
        self.treatment_range = treatment_range
        self.stratified_subsampling = stratified_subsampling
        self.bin_edges = bin_edges
        self.positivity_violations_dict = positivity_violations_dict
        self.seed=seed

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        """
        Estructura:
        -- recibo dataset:
        -- separo en exitos y fracasos
            --  exitos:
                propagan hacia arriba
                interpolan hacia abajo
            -- fracasos
                propagan hacia abajo
                interpolan hacia arriba

        """
        assert (
            self.interpolation_method in self.interpolation_methods
        ), f"Only {self.interpolation_methods} are currently supported for interpolation"

        data = pd.concat([X, y], axis=1)

        if "sample_weights" not in data.columns:
            data["sample_weights"] = 1

        success = data[data[self.outcome] == 1]
        failure = data[data[self.outcome] == 0]

        # exitos propagados
        success_prop = self.propagate(success, "greater")
        # fracasos propagados
        failure_prop = self.propagate(failure, "less")
        
        data_extrapolation = pd.concat(
            [success_prop, failure_prop], ignore_index=False
        )
        data_extrapolation = data_extrapolation[
            (data_extrapolation[self.treatment] >= self.treatment_range[0])
            & (data_extrapolation[self.treatment] <= self.treatment_range[1])
        ]
        data_extrapolation = filter_non_positivity_samples(data_extrapolation, self.positivity_violations_dict, True)
        if self.undersampling_perc_extrapolation:
            
            if self.stratified_subsampling:
                bins = pd.cut(data[self.treatment], bins=self.bin_edges, labels=False,)
                ref_proportions = pd.value_counts(bins, normalize=True)
                sample_bins = pd.qcut(data_extrapolation[self.treatment], len(ref_proportions), labels=False)

                # Calculate sample sizes for each bin
                target_size = int(len(data_extrapolation) * (1 - self.undersampling_perc_extrapolation))
                bin_samples = (ref_proportions * target_size).round().astype(int)

                # Sample from each bin and concatenate
                data_extrapolation = pd.concat([
                    data_extrapolation[sample_bins == bin].sample(
                        n=min(n, len(data_extrapolation[sample_bins == bin])),
                        random_state=self.seed
                    ) for bin, n in bin_samples.items()
                ])
            else:
                data_extrapolation = data_extrapolation.sample(
                    frac=1 - self.undersampling_perc_extrapolation,
                    random_state=self.seed
                )
        
        # concat
        output = pd.concat([data, data_extrapolation], ignore_index=False)

        if self.interpolation_method:
            if self.interpolation_method == self.interpolation_methods[1]:
                # exitos interpolados
                success_inter = self.linearly_interpolate_data(success, "smaller")
                # fracasos interpolados
                failure_inter = self.linearly_interpolate_data(failure, "greater")
            elif self.interpolation_method == self.interpolation_methods[2]:
                # exitos interpolados
                success_inter = self.quadratically_interpolate_data(success, "smaller")
                # fracasos interpolados
                failure_inter = self.quadratically_interpolate_data(
                    failure,
                    "greater"
                )
            data_interpolation = pd.concat(
                [success_inter, failure_inter], ignore_index=False
            )
            data_interpolation = data_interpolation[
                (data_interpolation[self.treatment] >= self.treatment_range[0])
                & (data_interpolation[self.treatment] <= self.treatment_range[1])
            ]
            data_interpolation = filter_non_positivity_samples(data_interpolation, self.positivity_violations_dict, True)
            if self.undersampling_perc_interpolation:
                np.random.seed(self.seed)
                if self.stratified_subsampling:
                    bins = pd.cut(data[self.treatment], bins=self.bin_edges, labels=False,)
                    ref_proportions = pd.value_counts(bins, normalize=True)
                    sample_bins = pd.qcut(data_interpolation[self.treatment], len(ref_proportions), labels=False)

                    # Calculate sample sizes for each bin
                    target_size = int(len(data_interpolation) * (1 - self.undersampling_perc_interpolation))
                    bin_samples = (ref_proportions * target_size).round().astype(int)

                    # Sample from each bin and concatenate
                    data_interpolation = pd.concat([
                        data_interpolation[sample_bins == bin].sample(
                            n=min(n, len(data_interpolation[sample_bins == bin])),
                            random_state=self.seed,
                            weights=data_interpolation[sample_bins == bin]["sample_weights"]  # Add weights here
                        ) for bin, n in bin_samples.items()
                    ])
                else:
                    data_interpolation = data_interpolation.sample(
                        frac=1 - self.undersampling_perc_interpolation,
                        random_state=self.seed,
                        weights=data_interpolation["sample_weights"]
                    )
                
            data_interpolation["sample_weights"] = pd.to_numeric(data_interpolation["sample_weights"])
            
            # remove redundant cases
            data_interpolation = self.correct_interpolated_treatment_extremes(data_interpolation)
            
            # concat
            output = pd.concat(
                [output, data_interpolation], axis=0, ignore_index=False
            )
        output[self.treatment] = pd.to_numeric(output[self.treatment])

        return output.drop(self.outcome, axis=1), output[[self.outcome]]

    def propagate(self, data, symbol):
        data["perc_steps"] = data.apply(
            lambda x: self._generate_perc_steps(
                self.perc_step_extrapolation, x[self.treatment], symbol, random_seed=x.name
            ),
            axis=1,
        )
        output = (
            data.drop(self.treatment, axis=1)
            .explode("perc_steps", ignore_index=False)
            .rename(columns={"perc_steps": self.treatment})
            .dropna(subset=self.treatment)
        )
        return output

    def linearly_interpolate_data(
        self,
        data,
        symbol,
    ):
        data["perc_steps"] = data.apply(
            lambda x: self._generate_perc_steps(
                self.perc_step_interpolation, x[self.treatment], symbol, random_seed=x.name
            ),
            axis=1,
        )

        if symbol == "greater":
            data["slope"] = data.apply(
                lambda x: self._compute_slope(x[self.treatment], 100), axis=1
            )
        else:
            data["slope"] = data.apply(
                lambda x: self._compute_slope(0, x[self.treatment]), axis=1
            )

        df = data.explode("perc_steps", ignore_index=False)

        if symbol == "greater":
            df["success_probs"] = df["slope"] * (df["perc_steps"] - df[self.treatment])
        else:
            df["success_probs"] = df["slope"] * df["perc_steps"]
        df["failure_probs"] = 1 - df["success_probs"]

        successes = df.drop([self.treatment, "slope", "failure_probs"], axis=1)
        successes["sample_weights"] = (
            successes["sample_weights"] * successes["success_probs"]
        )
        successes.rename(columns={"perc_steps": self.treatment}, inplace=True)
        successes.drop(
            "success_probs", axis=1, inplace=True
        )
        successes[self.outcome] = 1

        failures = df.drop([self.treatment, "slope", "success_probs"], axis=1)
        failures["sample_weights"] = (
            failures["sample_weights"] * failures["failure_probs"]
        )
        failures.rename(columns={"perc_steps": self.treatment}, inplace=True)
        failures.drop(
            "failure_probs", axis=1, inplace=True
        )
        failures[self.outcome] = 0

        output = pd.concat([successes, failures], axis=0, ignore_index=False)

        return output[(output["sample_weights"] > 0) & (output["sample_weights"] < 1)]

    def quadratically_interpolate_data(self, data, symbol, step=10):
        data["perc_steps"] = data.apply(
            lambda x: self._generate_perc_steps(
                self.perc_step_interpolation, x[self.treatment], symbol, random_seed=x.name
            ),
            axis=1,
        )
        if symbol == "greater":
            data["success_probs"] = data.apply(
                lambda x: self.quadratic_interpolation(
                    treatments=[x[self.treatment], 100],
                    power=self.quadratic_power,
                    x_interpolation_values=x["perc_steps"],
                ),
                axis=1,
            )
        else:
            data["success_probs"] = data.apply(
                lambda x: self.quadratic_interpolation(
                    treatments=[0, x[self.treatment]],
                    power=self.quadratic_power,
                    x_interpolation_values=x["perc_steps"],
                ),
                axis=1,
            )
        
        df = data.explode(["perc_steps", 'success_probs'], ignore_index=False)
        df['failure_probs'] = 1- df['success_probs']

        successes = df.drop([self.treatment, "failure_probs"], axis=1)
        successes["sample_weights"] = (
            successes["sample_weights"] * successes["success_probs"]
        )
        successes.rename(columns={"perc_steps": self.treatment}, inplace=True)
        successes.drop(
            "success_probs", axis=1, inplace=True
        )
        successes[self.outcome] = 1

        failures = df.drop([self.treatment, "success_probs"], axis=1)
        failures["sample_weights"] = (
            failures["sample_weights"] * failures["failure_probs"]
        )
        failures.rename(columns={"perc_steps": self.treatment}, inplace=True)
        failures.drop(
            "failure_probs", axis=1, inplace=True
        )
        failures[self.outcome] = 0

        output = pd.concat([successes, failures], axis=0, ignore_index=False)

        return output[(output["sample_weights"] > 0) & (output["sample_weights"] < 1)]

    def quadratic_interpolation(self, treatments, power, x_interpolation_values):
        # obtains probability or weights for points in the curve
        shift = treatments[0]

        def adjusted_quadratic_function(x, a):
            return a * (x - shift) ** power

        initial_guess_slopes = [0.001]
        params_adjusted_quadratic, _ = curve_fit(
            adjusted_quadratic_function,
            treatments,
            np.array([0, 1]),
            p0=initial_guess_slopes,
        )
        interpolated_values = adjusted_quadratic_function(
            np.array(x_interpolation_values), *params_adjusted_quadratic
        )

        return interpolated_values

    def correct_interpolated_treatment_extremes(self, data_int):
        # correct 0s and 100s with sample weights lower than 1
        data_int.loc[
            (data_int[self.treatment] == 0)
            & (data_int[self.outcome] == 0)
            & (data_int["sample_weights"] < 1),
            "sample_weights"
        ] = 1
        data_int.loc[
            (data_int[self.treatment] == 100)
            & (data_int[self.outcome] == 1)
            & (data_int["sample_weights"] < 1),
            "sample_weights"
        ] = 1
        
        # remove 0s and 100s with targets 1 and 0
        data_int = data_int[
            ~(
                (data_int[self.treatment] == 100)
                & (data_int[self.outcome] == 0)
            )
        ]
        data_int = data_int[
            ~(
                (data_int[self.treatment] == 0)
                & (data_int[self.outcome] == 1)
            )
        ]
        return data_int
        

    @staticmethod
    def _compute_slope(treatment_1, treatment_2):
        slope = 1 / (treatment_2 - treatment_1)
        return slope

    @staticmethod
    def adjusted_quadratic_function(x, a, shift, power):
        return a * (x - shift) ** power

    @staticmethod
    def _generate_perc_steps(step, quita, symbol, random_seed):
        random.seed(random_seed)
        extrapolation_steps = [
            max(
                min(
                    round(
                        random.uniform(number - step / 2, number + step / 2),
                        2,
                    ),
                    100,
                ),
                0,
            )
            for number in range(0, 100 + step, step)
        ]
        extrapolation_steps = list(set([0] + extrapolation_steps + [100]))
        rounded_quita = step * round(
            quita / step
        )  # this rounds quita to the center of the corresponding interval
        if symbol == "greater":
            row_steps = [v for v in extrapolation_steps if v > rounded_quita + step / 2]
        else:
            row_steps = [v for v in extrapolation_steps if v < rounded_quita - step / 2]

        return row_steps
    

def filter_non_positivity_samples(df_all_samples, treatment_ranges_dict,  keep_inside=False):
    """
    Filter samples based on specified ranges for each sample.
    """
    filtered_dfs = []
    for sample_idx, ranges in treatment_ranges_dict.items():
        # Get subset of DataFrame for current cluster
        try:
            sample_idx_df = df_all_samples.loc[[sample_idx], :].copy()
        except:
            continue

        if len(sample_idx_df) == 0:
            continue

        # Create boolean mask for values within any of the ranges
        in_range_mask = False
        for min_val, max_val in ranges:
            in_range_mask |= (sample_idx_df[TREATMENT].between(min_val, max_val))

        # Select samples based on keep_inside parameter
        selected_df = sample_idx_df[in_range_mask if keep_inside else ~in_range_mask]

        if len(selected_df) > 0:
            filtered_dfs.append(selected_df)
        
    
    # Combine all filtered DataFrames
    if filtered_dfs:
        return pd.concat(filtered_dfs, axis=0)
    else:
        return pd.DataFrame(columns=df_all_samples.columns)