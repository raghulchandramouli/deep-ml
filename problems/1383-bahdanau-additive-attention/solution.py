import numpy as np

def additive_attention(query, keys, W_q, W_k, v):
    # Step 1: Project query and keys
    q = W_q @ query          # (d_a,)
    k = keys @ W_k.T         # (n, d_a)

    # Step 2: Calculate attention scores
    e = np.tanh(q + k) @ v   # (n,)

    # Step 3: Softmax
    e = e - np.max(e)        # Numerical stability
    exp_e = np.exp(e)
    weights = exp_e / np.sum(exp_e)

    # Step 4: Weighted sum of keys
    context = weights @ keys  # (d_k,)

    return weights, context