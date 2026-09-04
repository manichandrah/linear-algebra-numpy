#Exercise 1 — rows 0, 3, 7. Plain single-array fancy indexing along axis 0 (§2.1). Shape(3, 10) .
import numpy as np
input_array = np.arange(100).reshape(10,10)
print(input_array[[0,3,7]])
