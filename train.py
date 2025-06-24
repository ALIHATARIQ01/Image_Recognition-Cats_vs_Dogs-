import pickle
import numpy as np
from model import NeuralNet
from config import EPOCHS, INITIAL_LR, BATCH_SIZE, MODEL_PATH

def train_model(X_train, y_train, X_test, y_test):
    model = NeuralNet(X_train.shape[1])
    for epoch in range(EPOCHS):
        lr = INITIAL_LR * (0.98 ** epoch)
        model.set_lr(lr)
        for i in range(0, len(X_train), BATCH_SIZE):
            Xb = X_train[i:i+BATCH_SIZE]
            yb = y_train[i:i+BATCH_SIZE]
            model.forward(Xb, training=True)
            model.backward(Xb, yb)
        out = model.forward(X_train)
        loss = -np.mean(y_train * np.log(out + 1e-8) + (1 - y_train) * np.log(1 - out + 1e-8))
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | Train Acc: {model.accuracy(X_train, y_train):.2%} | Test Acc: {model.accuracy(X_test, y_test):.2%}")
    return model

def predict_image(model, image_path, img_size):
    import cv2
    import numpy as np
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("[ERROR] Cannot load image.")
        return
    img = cv2.resize(img, img_size)
    img = img / 255.0
    flat = img.flatten().reshape(1, -1)
    pred = model.predict(flat)
    print(f"[RESULT] This is a {'DOG' if pred[0][0] else 'CAT'}.")
