#Exercise 5 — Create a 10×10 matrix using arange + reshape . Print the diagonal.
import numpy as np
matrix=np.arange(100).reshape(10, 10)
print(f"the matrix is \n{matrix}")
print(f"the diagonal is {np.diag(matrix)}")
