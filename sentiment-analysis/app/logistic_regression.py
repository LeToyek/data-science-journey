import numpy as np
from tqdm import tqdm


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iter=1000, verbose=False):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.verbose = verbose

    def _sigmoid(self, z) -> float:
        return 1 / (1 + np.exp(-z))

    def _loss(self, y,y_pred) -> float:
        return -np.mean(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.weights = np.zeros(self.n)
        self.bias = 0
      
        print('Training...')
        iterator = tqdm(range(self.n_iter),desc='Training', disable=not self.verbose)
        for i in iterator:
            z = np.dot(X, self.weights) + self.bias
            y_hat = self._sigmoid(z)
            # Compute gradients
            dw = (1 / self.m) * np.dot(X.T, (y_hat - y))
            db = (1 / self.m) * np.sum(y_hat - y)
            
            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if self.verbose:
                loss = self._loss(y,y_hat)
                iterator.set_postfix(loss=loss, iteration=i+1)
    
    def predict(self,X):
        z = np.dot(X, self.weights) + self.bias
        y_hat = self._sigmoid(z)
        y_pred = [1 if i > 0.5 else 0 for i in y_hat]
        return np.array(y_pred)