#Exercise 2 — Create a 5×5 identity matrix. Print shape, dtype, total bytes.
import numpy as np
print(f"5x5 Identity Matrix is\n{np.eye(5)}\n Data Type is {np.eye(5).dtype}\nShape is {np.eye(5).shape}\nTotal Space Occupied is{np.eye(5).nbytes}")
'''While your code is functionally correct and successfully outputs the right matrix and memory sizes, this specific styling is generally discouraged in standard software development.

Here is an objective breakdown of why this style is problematic and how to write it more idiomatically.
Issues with the Current Style
Violates the DRY Principle (Don't Repeat Yourself): You execute the exact same operation—np.full((4, 4), 7.0, dtype = np.float32)—twice. This forces the computer to allocate memory and build the array two separate times, wasting computational resources.
Poor Readability: F-strings are designed for string interpolation, not for housing complex logic or multi-line function calls. Packing heavy computations inside {} makes the code difficult to read and debug.
Maintenance Risks: Because the array creation is hardcoded three separate times across the print statements, changing the matrix size later means you have to hunt down and update every instance.
Inconsistencies: Because the logic is buried in the string, it is easy to miss typos. For example, your text string says "7x7 Matrix" while your code generates a (4, 4) matrix.

The Idiomatic Approach
The standard practice in Python is to separate data creation (logic) from data display (presentation). You do this by assigning your results to variables first.

Python
import numpy as np

# 1. Logic: Create the arrays and store them in variables
matrix_32 = np.full((4, 4), 7.0, dtype=np.float32)
matrix_64 = np.full((4, 4), 7.0, dtype=np.float64)

# 2. Presentation: Use the variables in your f-strings
print(f"4x4 Matrix filled with 7.0 having data type as float32 is\n{matrix_32}")
print(f"Space for float32 is {matrix_32.nbytes} bytes")
print(f"Space for float64 is {matrix_64.nbytes} bytes")
Why this is better:
Efficiency: The float32 matrix is only generated once.

Clarity: The print statements are clean, short, and easy to read.

Flexibility: If you want to change the matrix to 5x5 or the fill value to 8.0, you only have to change it in one highly visible place at the top of the code.'''
