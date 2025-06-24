import numpy as np

DATA_DIR = 'data/train'
IMG_SIZE = (64, 64)
EPOCHS = 100
INITIAL_LR = 0.01
BATCH_SIZE = 32
SEED = 42
MODEL_PATH = "model.pkl"

np.random.seed(SEED)
