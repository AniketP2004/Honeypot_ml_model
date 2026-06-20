import numpy as np

class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._prior = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y==c]
            self._mean[idx, :] = X_c.mean(axis = 0)
            self._var[idx, :] = X_c.var(axis = 0)
            self._prior[idx] = X_c.shape[0]/float(n_samples)


    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._prior[idx])
            posterior = np.sum(np.log(np.clip(self._pdf(idx, x), 1e-9, None)))
            posterior = posterior + prior 
            posteriors.append(posterior)

        # Returns the class label with highest posterior
        return self._classes[np.argmax(posteriors)]


    def _pdf(self, idx, x):
        mean = self._mean[idx]
        var = self._var[idx, :] + 1e-9
        numerator = np.exp(-((x - mean)**2)/(2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator/denominator
    