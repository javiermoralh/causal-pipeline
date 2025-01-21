import matplotlib.pyplot as plt
import seaborn as sns

def plot_hist(data, x):
    sns.histplot(data=data, x=x, bins=30)
    plt.figure(figsize=(10, 8))
    plt.show()