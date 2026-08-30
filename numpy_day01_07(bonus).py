#Exercise 7 (bonus) — Create arrays using each of these random functions. Print shape and 3 sample values.
import numpy as np
rng = np.random.default_rng()
array = rng.random((1, 3))
print(f"array is {array}")
print(f"shape is {array.shape}")
