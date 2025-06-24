import os
import cv2
import numpy as np
from tqdm import tqdm
from config import DATA_DIR, IMG_SIZE

def augment_image(img):
    flipped = cv2.flip(img, 1)
    angle = 15
    M = cv2.getRotationMatrix2D((IMG_SIZE[0]//2, IMG_SIZE[1]//2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, IMG_SIZE)
    return [img, flipped, rotated]

def load_images():
    X, y = [], []
    mean_intensity = {'cat': [], 'dog': []}
    cats_dir = os.path.join(DATA_DIR, 'cats')
    dogs_dir = os.path.join(DATA_DIR, 'dogs')

    for label, folder in enumerate([cats_dir, dogs_dir]):
        folder_name = 'cat' if label == 0 else 'dog'
        for file in tqdm(os.listdir(folder), desc=f"Loading {folder_name}s"):
            path = os.path.join(folder, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv2.resize(img, IMG_SIZE)
            img = img / 255.0
            aug_imgs = augment_image(img)
            for aimg in aug_imgs:
                mean_intensity[folder_name].append(np.mean(aimg))
                X.append(aimg.flatten())
                y.append(label)
    return np.array(X), np.array(y).reshape(-1, 1), mean_intensity
