#Exercise 6 — Create an array [1, 2, 3] as int32, then convert to float64. Verify the dtype changed.
import numpy as np
array=np.array([1, 2, 3], dtype=np.int32)
array2=array.astype(np.float64)
print(f"the array is {array} and dtype of array is {array.dtype}")
print(f"the array is {array2} and dtype of this array is {array2.dtype}")
