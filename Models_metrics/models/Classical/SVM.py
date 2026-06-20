import numpy as np

class SVM:
    def __init__(self, learning_rate= 0.001, lambda_param = 0.01, n_iters = 1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)
        
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                
                else: 
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.weights) - self.bias
        return np.sign(approx)
    

class MulticlassSVM:
    def __init__(self, learning_rate=0.001, lamdba_param = 0.01, n_iters = 1000):
        self.lr = learning_rate
        self.lambda_param = lamdba_param
        self.n_iters = n_iters
        self.classes = None
        self.svms = {}
        

    def fit(self, X, y):
        n_samples , n_features = X.shape

        self.classes = np.unique(y)

        for c in self.classes:
            y_binary = np.where(y==c, 1, -1)
            svm = SVM(self.lr, self.lambda_param, self.n_iters)
            svm.fit(X, y_binary)
            self.svms[c] = svm
        
        self.weights = np.zeros(n_features)
        self.bias = 0


    def predict(self, X):
        scores = []

        for c in self.classes:
            svm = self.svms[c]
            score = np.dot(X, svm.weights) - svm.bias
            scores.append(score)

        stacked = np.stack(scores, axis=1)
        highest_score = np.argmax(stacked, axis=1)
        return self.classes[highest_score]

