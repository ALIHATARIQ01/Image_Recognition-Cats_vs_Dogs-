
# 🐱🐶 Cat vs Dog Image Classifier (from Scratch using NumPy & OpenCV)

This project is a **from-scratch implementation** of a binary image classifier that distinguishes between cats and dogs using a fully connected neural network. It uses **only NumPy and OpenCV**, with **no ML libraries** like TensorFlow or PyTorch.

It includes:
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
  - 🐱 200+ cat images
  - 🐶 200+ dog images

All images are:
- Grayscale
- Resized to **64×64**
- Augmented using horizontal flip and slight rotation

### 📁 Folder Structure

```
data/
├── train/
│   ├── cats/   # 1000 cat training images (e.g., 1.png, 2.png, ...)
│   └── dogs/   # 1000 dog training images
└── test/
    ├── cats/   # 200+ cat test images
    └── dogs/   # 200+ dog test images
```

> ✅ Make sure all images are `.png` and valid.  
> ❌ Corrupt or unreadable images are automatically skipped.

---

## 📁 Project Structure

```
Image_recognition/
├── main.py                  # CLI interface: train or predict
├── config.py                # Configuration variables (paths, image size, etc.)
├── data_loader.py           # Loads and augments training images
├── analysis.py              # Correlation and box plot visualizations
├── model.py                 # Manual neural network implementation
├── train.py                 # Training loop and prediction function
├── utils.py                 # Train/test split utility
├── model.pkl                # Saved model (after training)
├── boxplot.png              # Saved box plot of mean intensities
├── Cats_correlation.png     # Correlation heatmap for cats
├── Dogs_correlation.png     # Correlation heatmap for dogs
└── data/                    # Your dataset folder
```

---

## 🛠 Installation

Install the required Python packages:

```bash
pip install numpy opencv-python matplotlib seaborn tqdm
```

---

## 🚀 How to Run

### 🏋️‍♀️ Train the Model

```bash
python main.py
```

- Choose `train` when prompted.  
This will:

- Load and augment data from `data/train`
- Perform cross-correlation and box plot analysis
- Train a fully connected neural network from scratch  
- Save:
  - Model to `model.pkl`
  - Visualizations:
    - `boxplot.png`
    - `Cats_correlation.png`
    - `Dogs_correlation.png`

---

### 🔍 Predict a Single Image

```bash
python main.py
```

- Choose `predict` when prompted.
- Enter the path to a `.png` test image, for example:

```bash
data/test/dogs/45.png
```

**Output:**

```
[RESULT] This is a DOG.
```

---

## 🧠 Neural Network Architecture

Implemented manually in `model.py`:

- **Input**: 4096 features (64×64 grayscale)
- **Hidden Layer 1**: 256 neurons (ReLU + dropout)
- **Hidden Layer 2**: 128 neurons (ReLU)
- **Hidden Layer 3**: 64 neurons (ReLU)
- **Output**: 1 neuron (Sigmoid for binary classification)

**Other details:**
- **Loss function**: Binary Cross-Entropy
- **Optimizer**: Manual Gradient Descent
- **Learning rate decay**:  
  `lr = INITIAL_LR * (0.98 ** epoch)`

---

## 📊 Output Visualizations

Automatically saved after training:

- `boxplot.png` – Distribution of mean pixel intensities for cats vs dogs
- `Cats_correlation.png` – Heatmap showing similarity between 10 sample cat images
- `Dogs_correlation.png` – Heatmap showing similarity between 10 sample dog images

---

## 💡 Highlights

✅ Written entirely with NumPy & OpenCV  
✅ Works on grayscale `.png` images  
✅ Visual analytics (boxplot, correlation matrix)  
✅ No deep learning libraries used  


✅ Lightweight and educational

---
## 👩‍💻 Author
- Aliha Tariq
- Computer Scientist & Developer
- Passionate about AI, Computer Vision, and Clean Code
