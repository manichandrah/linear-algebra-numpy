#Exercise 10 — Clock the difference. Time summing 2,000,000 numbers, list-style vs array-style.
import time
import numpy as np
my_list = list(range(2000000))
array = np.arange(2000000)
start_list = time.perf_counter()
sum(my_list)
end_list = time.perf_counter()
list_time = end_list - start_list

start_array = time.perf_counter()
sum(array)
end_array = time.perf_counter()
array_time = end_array - start_array
print(f"Python list took: {list_time:.5f}")
print(f"Python Array took: {array_time:.5f}")
