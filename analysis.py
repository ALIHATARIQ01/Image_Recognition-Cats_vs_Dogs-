import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def pearson_corr(img1, img2):
    i1, i2 = img1.flatten(), img2.flatten()
    mean1, mean2 = np.mean(i1), np.mean(i2)
    num = np.sum((i1 - mean1) * (i2 - mean2))
    den = np.sqrt(np.sum((i1 - mean1)**2) * np.sum((i2 - mean2)**2))
    return num / den if den != 0 else 0

def cross_correlation_matrix(images, label):
    sample = images[:10]
    matrix = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            matrix[i][j] = pearson_corr(sample[i], sample[j])
    sns.heatmap(matrix, cmap="coolwarm")
    plt.title(f"{label} Cross-Correlation")
    plt.savefig(f"{label}_correlation.png")
    plt.close()

def plot_boxplot(mean_intensity):
    plt.figure()
    sns.boxplot(data=[mean_intensity['cat'], mean_intensity['dog']])
    plt.xticks([0, 1], ['Cats', 'Dogs'])
    plt.ylabel('Mean Intensity')
    plt.title('Box Plot - Mean Intensities')
    plt.savefig("boxplot.png")
    plt.close()
