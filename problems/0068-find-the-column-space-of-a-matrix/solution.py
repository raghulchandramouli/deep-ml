import numpy as np

def matrix_image(A):
    A = A.astype(float)
    rows, cols = A.shape
    R = A.copy()
    pivot_cols = []
    row = 0

    for col in range(cols):
        if row >= rows:
            break

        # Find a row with a non-zero entry in this column
        for r in range(row, rows):
            if R[r, col] != 0:
                R[[row, r]] = R[[r, row]]  # Swap
                break
        else:
            continue  # No pivot in this column

        R[row] /= R[row, col]  # Make pivot 1

        # Eliminate entries below the pivot
        for r in range(row + 1, rows):
            R[r] -= R[r, col] * R[row]

        pivot_cols.append(col)  # Track pivot column index
        row += 1

    # Return original matrix's pivot columns
    return A[:, pivot_cols]
