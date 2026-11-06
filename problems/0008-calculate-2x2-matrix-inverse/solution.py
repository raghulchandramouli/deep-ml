import numpy

def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    """
    Calculate the inverse of a 2x2 matrix.
    
    Args:
        matrix: A 2x2 matrix represented as [[a, b], [c, d]]
    
    Returns:
        The inverse matrix as a 2x2 list, or None if the matrix is singular
        (i.e., determinant equals zero)
    """

    a, b = matrix[0]
    c, d = matrix[1]

    determinant = a * d - c * b
    if determinant == 0:
        return None
    # Your code here
    inverse = numpy.linalg.inv(matrix)
    return inverse