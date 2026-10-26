import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	X, y  = np.array(X), np.array(y)
	x_t_x = np.dot(X.T, X)
	x_1   = np.linalg.inv(x_t_x)
	x_t_y = np.dot(X.T, y)

	 
	return np.dot(x_1, x_t_y)