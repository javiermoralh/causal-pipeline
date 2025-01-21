import numpy as np

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

