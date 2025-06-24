import os
import pickle
from config import MODEL_PATH, IMG_SIZE
from data_loader import load_images
from analysis import cross_correlation_matrix, plot_boxplot
from utils import train_test_split
from train import train_model, predict_image

def main():
    choice = input("Do you want to (train/predict)? ").strip().lower()
    if choice == 'train':
        X, y, intensity = load_images()
        cross_correlation_matrix(X[y.flatten() == 0], 'Cats')
        cross_correlation_matrix(X[y.flatten() == 1], 'Dogs')
        plot_boxplot(intensity)
        X_train, y_train, X_test, y_test = train_test_split(X, y)
        model = train_model(X_train, y_train, X_test, y_test)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        print("[INFO] Model trained and saved.")
    elif choice == 'predict':
        if not os.path.exists(MODEL_PATH):
            print("[ERROR] Model not found. Train first.")
            return
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        image_path = input("Enter image path to predict: ").strip()
        predict_image(model, image_path, IMG_SIZE)
    else:
        print("[ERROR] Invalid choice. Please enter 'train' or 'predict'.")

if __name__ == "__main__":
    main()

print("hello")