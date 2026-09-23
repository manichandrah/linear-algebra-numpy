#Exercise 5: Count how many values are positive in a 100-element random array.
import numpy as np
rng = np.random.default_rng()
random_array = rng.random(100)
bool_mask = random_array > 0
positive_count = random_array[bool_mask]
print(len(positive_count))
random_array_2 = rng.standard_normal(100)
bool_mask_2 = random_array_2 > 0
positive_count_2 = random_array_2[bool_mask_2]
print(len(positive_count_2))
