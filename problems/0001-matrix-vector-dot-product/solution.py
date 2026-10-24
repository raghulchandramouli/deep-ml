import numpy as np

def matrix_dot_vector(a: list[list[int | float]], b: list[int | float]) -> list[int | float]:
    if any(len(row) != len(b) for row in a):
        return -1

    return (np.array(a) @ np.array(b)).tolist()