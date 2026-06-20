import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None 
        self.classes = None

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)

        # one hot encoding
        y_encoded = np.zeros((n_samples, n_classes))
        for idx, c in enumerate(self.classes):
            y_encoded[:, idx] = (y ==c).astype(int)

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._softmax(linear_model)

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y_encoded))
            db = (1/n_samples) * np.sum(y_pred - y_encoded, axis=0)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._softmax(linear_model)
        return self.classes[np.argmax(y_pred, axis=1)]
    