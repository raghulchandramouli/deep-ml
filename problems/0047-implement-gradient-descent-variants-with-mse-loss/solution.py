
import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    m, n = X.shape
    weights = np.asarray(weights, dtype=float).copy()
    y = np.asarray(y).flatten()

    if method == 'batch':
        batch_size = m
    elif method == 'stochastic':
        batch_size = 1
    elif method == 'mini_batch':
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
    else:
        raise ValueError("Invalid method")

    for epoch in range(n_epochs):
        for start in range(0, m, batch_size):
            X_batch = X[start:start + batch_size]
            y_batch = y[start:start + batch_size]

            y_hat = X_batch @ weights
            e = y_hat - y_batch

            g = (2 / len(X_batch)) * (X_batch.T @ e)

            weights -= learning_rate * g

    return weights
