import numpy as np

class NeuralNet:
    def __init__(self, input_size):
        self.W1 = np.random.randn(input_size, 256) / np.sqrt(input_size)
        self.b1 = np.zeros((1, 256))
        self.W2 = np.random.randn(256, 128) / np.sqrt(256)
        self.b2 = np.zeros((1, 128))
        self.W3 = np.random.randn(128, 64) / np.sqrt(128)
        self.b3 = np.zeros((1, 64))
        self.W4 = np.random.randn(64, 1) / np.sqrt(64)
        self.b4 = np.zeros((1, 1))

    def sigmoid(self, z): return 1 / (1 + np.exp(-z))
    def relu(self, z): return np.maximum(0, z)
    def relu_deriv(self, z): return (z > 0).astype(float)

    def forward(self, X, training=False):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)
        if training: self.A1 *= (np.random.rand(*self.A1.shape) > 0.2)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.relu(self.Z2)
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = self.relu(self.Z3)
        self.Z4 = self.A3 @ self.W4 + self.b4
        self.A4 = self.sigmoid(self.Z4)
        return self.A4

    def backward(self, X, Y):
        m = Y.shape[0]
        dZ4 = self.A4 - Y
        dW4 = self.A3.T @ dZ4 / m
        db4 = np.sum(dZ4, axis=0, keepdims=True) / m

        dA3 = dZ4 @ self.W4.T
        dZ3 = dA3 * self.relu_deriv(self.Z3)
        dW3 = self.A2.T @ dZ3 / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * self.relu_deriv(self.Z2)
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_deriv(self.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W4 -= self.lr * dW4
        self.b4 -= self.lr * db4
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict(self, X): return (self.forward(X) > 0.5).astype(int)
    def accuracy(self, X, Y): return np.mean(self.predict(X) == Y)
    def set_lr(self, lr): self.lr = lr
