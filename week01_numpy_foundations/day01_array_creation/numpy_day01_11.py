#Exercise 11 — Prove the loop trap. The important one. Compare a manual loop over an array to .sum() on the same array.
import numpy as np
import time
numpy_array = np.arange(2000000)
for_array_start = time.perf_counter()
total = 0
for x in numpy_array:
    total+=x
for_array_end = time.perf_counter()
numpy_array_start = time.perf_counter()
sum_numpy = np.sum(numpy_array)
numpy_array_end = time.perf_counter()

for_loop_time = for_array_end - for_array_start
numpy_time = numpy_array_end - numpy_array_start

print(f"time taken for for loop is {for_loop_time:.5f} seconds")
print(f"time taken for numpy sum is {numpy_time:.5f}seconds")
