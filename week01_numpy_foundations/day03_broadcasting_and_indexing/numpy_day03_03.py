#Exercise 3: Use np.ix_ to extract a submatrix.
import numpy as np
matrix = np.arange(100).reshape(10,10)
rows = [0, 5]
cols = [5, 0]
ix_rows, ix_cols = np.ix_(rows, cols)
submatrix = matrix[np.ix_(rows, cols)]#equivalent to matrix[ix_rows, ix_cols]
print(submatrix)
