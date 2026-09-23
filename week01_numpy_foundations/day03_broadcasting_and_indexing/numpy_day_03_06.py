#Exercise 6: Given student scores (10 students × 5 subjects), find students who scored >80 in ALL subjects.
import numpy as np
rng = np.random.default_rng()
student0 = rng.integers(0, 100, size=5)
student1 = rng.integers(0, 100, size=5)
student2 = rng.integers(0, 100, size=5)
student3 = rng.integers(0, 100, size=5)
student4 = rng.integers(0, 100, size=5)
student5 = rng.integers(0, 100, size=5)
student6 = rng.integers(0, 100, size=5)
student7 = rng.integers(0, 100, size=5)
student8 = rng.integers(0, 100, size=5)
student9 = rng.integers(0, 100, size=5)
for x in [student0, student1, student2, student3, student4, student5, student6, student7, student8, student9]:
    bool_mask = x > 80
    print(x[bool_mask])


'''thats what, tell me how to iterate them in the list first, i already know that i have to iterate them in a list first

Ah, I understand now! You want to know the syntax to dynamically grab the variables student0, student1, etc., using a loop counter n so you don't have to type out all 10 variable names manually.

Because Python treats variable names as memory references, not text, you cannot just type student + n. To dynamically access a variable by its text name, you have to look it up in Python's internal dictionary of active variables using the locals() or globals() function.

Here is the exact syntax to do what you are asking.

1. The Syntax to Build the List Dynamically
If you want to construct the list first (which matches the [n:studentn] logic you were aiming for), you can use a list comprehension combined with an f-string and locals():

Python
# This dynamically creates a list containing the actual arrays
# by looking up "student0", "student1", etc.
all_students = [locals()[f"student{n}"] for n in range(10)]

# Now you can iterate normally
for x in all_students:
    bool_mask = x > 80
    print(bool_mask)
2. The Syntax to Iterate Directly
If you want to skip building the list and just loop through the range of numbers to grab the variables on the fly, you do this:

Python
for n in range(10):
    # Dynamically build the string name (e.g., "student0")
    var_name = f"student{n}"
    
    # Grab the actual variable from Python's memory
    current_student = locals()[var_name]
    
    # Apply your logic
    bool_mask = current_student > 80
    print(f"Mask for {var_name}: {bool_mask}")
Why locals()?
When you define variables in a script, Python stores them in a hidden dictionary. locals() lets you access that dictionary. By using an f-string f"student{n}", you are telling Python: "Generate the string 'student0', and then go fetch whatever data is stored under that name in your dictionary."

(Note: If your variables are defined at the very top level of your script outside of any function, globals() works exactly the same way as locals() here).'''
