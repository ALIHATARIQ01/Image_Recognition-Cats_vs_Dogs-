# 🐱🐶 Cat vs Dog Image Classifier (from Scratch using NumPy & OpenCV)

This project is a **from-scratch implementation** of a binary image classifier that distinguishes between cats and dogs using a fully connected neural network. It uses **only NumPy and OpenCV**, with **no ML libraries** like TensorFlow or PyTorch. Additionally, it includes:

- Manual forward and backward propagation  
- Image augmentation (flip + rotation)  
- Statistical visualizations (box plots, correlation heatmaps)  
- Model saving and loading via `pickle`

---

## 🗂 Dataset Description

- **Training set**:
  - 🐱 1000 cat images
  - 🐶 1000 dog images
- **Testing set**:
  - 🐱 100+ cat images
  - 🐶 100+ dog images

All images are:
- Grayscale
- Resized to **64×64**
- Augmented using horizontal flip and slight rotation

### 📁 Folder Structure

data/
├── train/
│ ├── cats/ # 1000 cat training images (e.g., 1.png, 2.png, ...)
│ └── dogs/ # 1000 dog training images
└── test/
├── cats/ # 100+ cat test images
└── dogs/ # 100+ dog test images
