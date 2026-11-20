import numpy as np

def simple_self_attention(X):
    # Convert list to NumPy array
    X = np.asarray(X, dtype=float)

    # Step 1: Attention scores
    scores = X @ X.T

    # Step 2: Stable softmax
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Step 3: Context vectors
    Z = weights @ X

    return Z.tolist()