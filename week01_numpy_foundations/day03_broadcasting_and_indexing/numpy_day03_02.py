#Exercise 2: Create a 1D array of 100 random values from Normal(0, 1). Select all values between -1 and 1.
import numpy as np
rng = np.random.default_rng()
random_array = rng.standard_normal(100)
print(random_array)
values = (random_array>-1) & (random_array<1)
print(values)
print(random_array[values])
print(f"{(len(random_array[values])/len(random_array)*100)}")
