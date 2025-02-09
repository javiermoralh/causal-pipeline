import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cumgain_curve(df: pd.DataFrame, ite_col: str='ite') -> tuple:
    df = df.copy()
    df['treatment_rank'] = df['ite'].rank(ascending=False)
    df = df.sort_values('treatment_rank')

    # Compute cumulative sum of outcomes
    df['cumulative_gain'] = df['ite'].cumsum() / df['ite'].abs().sum()

    # Compute percentiles
    df['percentile'] = np.linspace(1, 100, len(df)) / 100
    return df.percentile.values, df.cumulative_gain.values


def plot_cumgain(percentiles: np.array, cumgains: np.array, ax=None) -> None:
    if ax is not None:
        ax.plot(percentiles, cumgains, label="Cumulative gain curve")
        ax.plot(percentiles, percentiles, '--', label="random")
        ax.set_xlabel("Cumulative Percentage of Population")
        ax.set_ylabel("cummulative gain")
        ax.set_title("Cumulative Gain Curve")
    else:
        plt.plot(percentiles, cumgains, label="Cumulative gain curve")
        plt.plot(percentiles, percentiles, '--', label="random")
        plt.xlabel("Cumulative Percentage of Population")
        plt.ylabel("cummulative gain")
        plt.title("Cumulative Gain Curve")
        plt.legend()
        plt.show()

def cumgain_auc(percentiles: np.array, cumgains: np.array) -> float:
    return np.trapz(cumgains, percentiles)

