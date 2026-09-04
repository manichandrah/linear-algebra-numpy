#Exercise 9 — Weigh a list vs an array. Build 100,000 ints both ways and compare what each structure actually costs.
import sys
import numpy as np
my_list = [x for x in range(100000)]
array = np.arange(100000)
list_container = sys.getsizeof(my_list)
list_contents = sum(sys.getsizeof(x) for x in my_list)
list_weight = list_container + list_contents
print(f"space occupied by list carrying 100,000 elements is {list_weight}")
print(f"space occupied by array carrying 100,000 elements is {array.nbytes}")
