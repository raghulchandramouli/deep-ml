import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    # Your code here
    
    grad = np.zeros(x.shape)

    for idx in np.ndindex(x.shape):
        x_add = x.copy()
        x_add[idx] += epsilon

        x_minus = x.copy()
        x_minus[idx] -= epsilon

        grad[idx] = (f(x_add) - f(x_minus)) / (2 * epsilon)

        relative_error = (
            np.linalg.norm(grad - analytical_grad) 
            / (np.linalg.norm(grad) + np.linalg.norm(analytical_grad) + 1e-12)
        )

    return grad, relative_error
