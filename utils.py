import numpy as np

def train_test_split(X, y, ratio=0.2):
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(len(X) * (1 - ratio))
    return X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]]
