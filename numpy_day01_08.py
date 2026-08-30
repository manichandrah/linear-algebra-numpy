#Exercise 8 — Measure the object tax. Confirm a lone Python int costs more than the 8 raw bytes its value needs.
import sys
import numpy as np
python_int = 1
numpy_int = np.arange(3)
print(f"Python in costs about {sys.getsizeof(python_int)} bytes")
print(f"Numpy int costs about {numpy_int.itemsize} bytes")
