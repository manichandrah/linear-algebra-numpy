# 🔬 PHASE 1.1 — COMPLETE DAY-BY-DAY EXECUTION PLAN
## Mathematics & NumPy Foundation
### August 1 – October 31, 2026 (13 weeks / 78 working days)

> **How this document works:**
> - Every working day (Mon–Sat) has its own section
> - Sundays are rest/catch-up days — **do not skip rest**
> - Each day tells you: what to watch/read, what to code, what to push to GitHub
> - Every exercise has starter code or exact specifications so you never stare at a blank screen
> - ✅ checkboxes at the end of each day — don't move forward until all are checked
> - If a day takes longer than expected, extend into the next day. **Don't skip ahead.**
>
> **Your daily structure (non-negotiable):**
> ```
> 🌅 Morning Session   (6:00 – 7:30 AM)   →  Math: Watch/Read + Take handwritten notes
> 🏫 College           (8:30 – 3:30 PM)    →  Attend classes
> 🔧 Afternoon Session (4:00 – 5:00 PM)    →  DSA: 1 LeetCode problem (separate track)
> 🌙 Evening Session   (7:00 – 9:00 PM)    →  NumPy: Implement what you learned in the morning
> 📤 Night Wrap-up     (9:00 – 9:30 PM)    →  Push code to GitHub + write commit message explaining what you learned
> ```
>
> **GitHub Setup (do this BEFORE Day 1):**
> 1. Create a GitHub account if you don't have one
> 2. Create a repository: `math-and-numpy-from-scratch`
> 3. Inside it, create these folders:
>    ```
>    math-and-numpy-from-scratch/
>    ├── week01_numpy_foundations/
>    ├── week02_linear_algebra_part1/
>    ├── week03_linear_algebra_part1_contd/
>    ├── week04_linear_algebra_part2/
>    ├── week05_linear_algebra_part2_contd/
>    ├── week06_calculus/
>    ├── week07_calculus_contd/
>    ├── week08_probability/
>    ├── week09_probability_contd/
>    ├── week10_neural_network_from_scratch/
>    ├── week11_review_and_capstone/
>    ├── week12_buffer/
>    ├── week13_buffer/
>    └── README.md
>    ```
> 4. Write a README: "Learning math and NumPy from scratch. Implementing everything by hand."
> 5. Make your first commit. **The streak starts today.**

---
---

# 📦 MONTH 1: LINEAR ALGEBRA + NUMPY CORE
## August 1 – August 31, 2026

---

# WEEK 1 — NUMPY FOUNDATION SPRINT
### Aug 4 (Mon) – Aug 9 (Sat)
> *Pure NumPy. No math yet. Build the tool before using it.*
> **Source:** [Python Data Science Handbook Ch.2](https://jakevdp.github.io/PythonDataScienceHandbook/) + [From Python to NumPy (Rougier)](https://www.labri.fr/perso/nrougier/from-python-to-numpy/) Ch. 1-2

---

## 📅 Day 1 — Monday, Aug 4: Arrays — Creation & Attributes

### 🌅 Morning (6:00 – 7:30 AM) — Read & Learn

**What to read:** VanderPlas Ch. 2.1 ("Understanding Data Types in Python") + Ch. 2.2 ("The Basics of NumPy Arrays")
- URL: https://jakevdp.github.io/PythonDataScienceHandbook/02.01-understanding-data-types.html
- URL: https://jakevdp.github.io/PythonDataScienceHandbook/02.02-the-basics-of-numpy-arrays.html

**While reading, write notes on:**
- Why Python lists are slow (each element is a full Python object with type info, reference count, etc.)
- Why NumPy arrays are fast (contiguous block of memory, fixed type, no overhead per element)
- The difference between `dtype=np.float32` and `dtype=np.float64` (memory and precision)

**Key functions to memorize (write these on a sticky note on your monitor):**
```
np.array()          np.zeros()         np.ones()
np.full()           np.empty()         np.eye()
np.arange()         np.linspace()      np.logspace()
np.zeros_like()     np.ones_like()     np.identity()
np.diag()           np.random.rand()   np.random.randn()
```

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File to create:** `week01_numpy_foundations/day01_array_creation.py`

**Exercise 1:** Create a 1D array of 50 values linearly spaced between 0 and 2π. Print it.
```python
import numpy as np

# YOUR CODE: Use np.linspace
x = ___  # 50 values from 0 to 2*pi
print(f"Shape: {x.shape}, Dtype: {x.dtype}")
print(f"First 5 values: {x[:5]}")
print(f"Last value should be ~6.28: {x[-1]:.4f}")
```

**Exercise 2:** Create a 5×5 identity matrix. Print shape, dtype, total bytes.
```python
I = np.eye(5)
print(f"Shape: {I.shape}")        # Expected: (5, 5)
print(f"Dtype: {I.dtype}")        # Expected: float64
print(f"Total bytes: {I.nbytes}") # Expected: 200 (5*5*8 bytes for float64)
print(f"Diagonal: {np.diag(I)}")  # Expected: [1. 1. 1. 1. 1.]
```

**Exercise 3:** Create a 4×4 matrix filled with 7.0 as float32. Verify it uses half the memory of float64.
```python
a_32 = np.full((4, 4), 7.0, dtype=np.float32)
a_64 = np.full((4, 4), 7.0, dtype=np.float64)
print(f"float32 bytes: {a_32.nbytes}")  # Expected: 64
print(f"float64 bytes: {a_64.nbytes}")  # Expected: 128
print(f"Ratio: {a_64.nbytes / a_32.nbytes}x")  # Expected: 2.0
```

**Exercise 4:** Create a 3D array of shape (2, 3, 4) filled with zeros. Print ndim and size.
```python
z = np.zeros((2, 3, 4))
print(f"ndim: {z.ndim}")   # Expected: 3
print(f"size: {z.size}")   # Expected: 24 (2*3*4)
print(f"shape: {z.shape}") # Expected: (2, 3, 4)
```

**Exercise 5:** Create a 10×10 matrix using `arange` + `reshape`. Print the diagonal.
```python
M = np.arange(100).reshape(10, 10)
print(f"Matrix:\n{M}")
print(f"Diagonal: {np.diag(M)}")  # Expected: [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]
```

**Exercise 6:** Create an array `[1, 2, 3]` as int32, then convert to float64. Verify dtype changed.
```python
a = np.array([1, 2, 3], dtype=np.int32)
print(f"Before: dtype={a.dtype}")  # int32
b = a.astype(np.float64)
print(f"After: dtype={b.dtype}")   # float64
print(f"Values: {b}")              # [1. 2. 3.]
```

**Exercise 7 (bonus):** Create arrays using each of these random functions. Print shape and 3 sample values:
```python
rng = np.random.default_rng(42)  # Modern way — always seed for reproducibility

a = rng.random((3, 3))           # Uniform [0, 1)
b = rng.standard_normal((3, 3))  # Normal(0, 1)
c = rng.integers(0, 100, (3, 3)) # Random ints [0, 100)

# Print each with its shape
```

### 📤 Night (9:00 – 9:30 PM) — Push

```bash
cd math-and-numpy-from-scratch
git add week01_numpy_foundations/day01_array_creation.py
git commit -m "Day 1: NumPy array creation — np.array, zeros, ones, full, eye, arange, linspace, dtype, astype, nbytes"
git push
```

### ✅ Day 1 Checklist
- [ ] I can create a 1D, 2D, and 3D array without looking anything up
- [ ] I know the difference between float32 and float64 (memory and precision)
- [ ] I know `.shape`, `.ndim`, `.size`, `.dtype`, `.nbytes`, `.itemsize`
- [ ] I understand why `np.random.default_rng(seed)` is better than `np.random.rand()`
- [ ] Code is pushed to GitHub

---

## 📅 Day 2 — Tuesday, Aug 5: Indexing & Slicing

### 🌅 Morning (6:00 – 7:30 AM) — Read & Learn

**What to read:** VanderPlas Ch. 2.2 continued — focus on the "Array Indexing" and "Array Slicing" sections.
- Also read: Rougier "From Python to NumPy" — Section 2.3 (Slicing and Views)

**Critical concept to internalize:**
> **Slicing returns a VIEW, not a copy.** If you modify a slice, you modify the original array.
> This is the #1 source of bugs for NumPy beginners. Understand this deeply.

**Notation cheat sheet (write this down):**
```
a[start:stop:step]     # 1D slice
a[row_start:row_stop, col_start:col_stop]  # 2D slice
a[:, 0]                # Entire first column
a[0, :]                # Entire first row
a[::-1]                # Reverse
a[..., 0]              # Ellipsis — all dimensions except last, then index 0
a[:, np.newaxis]       # Add a new axis (dimension)
```

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week01_numpy_foundations/day02_indexing_slicing.py`

**Exercise 1:** Create a 6×6 matrix (values 0–35). Extract the 2×2 block from the bottom-right corner.
```python
M = np.arange(36).reshape(6, 6)
print("Full matrix:")
print(M)

# YOUR CODE: Extract bottom-right 2x2
bottom_right = M[___:___, ___:___]
print(f"\nBottom-right 2x2:\n{bottom_right}")
# Expected: [[28 29] [34 35]]
```

**Exercise 2:** Create a 1D array of 20 elements. Reverse it using slicing only.
```python
a = np.arange(20)
reversed_a = a[___]  # Use slicing, no function calls
print(f"Original:  {a}")
print(f"Reversed:  {reversed_a}")
# Verify: reversed_a[0] should equal a[-1]
assert reversed_a[0] == a[-1], "Reversal failed!"
```

**Exercise 3:** Create a 5×5 matrix. Extract every other row AND every other column.
```python
M = np.arange(25).reshape(5, 5)
# Extract rows 0, 2, 4 and columns 0, 2, 4
checkerboard = M[___:___:___, ___:___:___]
print(f"Checkerboard:\n{checkerboard}")
# Expected shape: (3, 3)
assert checkerboard.shape == (3, 3), f"Wrong shape: {checkerboard.shape}"
```

**Exercise 4: THE VIEW vs COPY DEMO — this is critical**
```python
# PART A: Demonstrate the VIEW problem
original = np.array([10, 20, 30, 40, 50])
my_slice = original[1:4]        # This is a VIEW — same memory!
print(f"Before modifying slice: original = {original}")

my_slice[0] = 999               # Modify the slice
print(f"After modifying slice:  original = {original}")
# SURPRISE: original is now [10, 999, 30, 40, 50] — it changed!

# PART B: Use .copy() to avoid this
original2 = np.array([10, 20, 30, 40, 50])
my_copy = original2[1:4].copy()  # This is an independent COPY
my_copy[0] = 999
print(f"\nWith .copy(): original2 = {original2}")
# original2 is still [10, 20, 30, 40, 50] — unchanged!
```
> **Write in your notes:** "SLICING = VIEW (shares memory). Use .copy() when I need independence."

**Exercise 5:** Create a (3, 4, 5) tensor. Use ellipsis to select where last axis = 0.
```python
T = np.arange(60).reshape(3, 4, 5)
print(f"Tensor shape: {T.shape}")

# Select all elements where last axis index is 0
result = T[..., 0]
print(f"Result shape: {result.shape}")  # Expected: (3, 4)
print(f"Result:\n{result}")
```

**Exercise 6:** Take a 1D array of shape (10,) and reshape it to (10,1), (1,10), and (1,10,1) using np.newaxis.
```python
a = np.arange(10)
print(f"Original shape: {a.shape}")        # (10,)

col_vector = a[:, np.newaxis]
print(f"Column vector: {col_vector.shape}")  # (10, 1)

row_vector = a[np.newaxis, :]
print(f"Row vector: {row_vector.shape}")     # (1, 10)

weird_shape = a[np.newaxis, :, np.newaxis]
print(f"3D shape: {weird_shape.shape}")      # (1, 10, 1)
```

**Exercise 7:** Extract the anti-diagonal of a 5×5 matrix (top-right to bottom-left).
```python
M = np.arange(25).reshape(5, 5)
# Hint: Flip the matrix horizontally first, then take np.diag
anti_diag = np.diag(M[:, ::-1])
print(f"Anti-diagonal: {anti_diag}")
# Expected: [4, 8, 12, 16, 20]
```

### ✅ Day 2 Checklist
- [ ] I can slice any 1D, 2D, or 3D array confidently
- [ ] I understand VIEW vs COPY and can demonstrate the difference
- [ ] I know what `np.newaxis` does and when to use it
- [ ] I know what `...` (ellipsis) does in indexing
- [ ] Code is pushed to GitHub

---

## 📅 Day 3 — Wednesday, Aug 6: Fancy Indexing & Boolean Indexing

### 🌅 Morning (6:00 – 7:30 AM) — Read

**What to read:** VanderPlas Ch. 2.7 ("Fancy Indexing") + Ch. 2.6 ("Comparisons, Masks, and Boolean Logic")

**Key distinction to write in your notes:**
```
SLICING:         a[1:5]        → Returns a VIEW (shares memory)
FANCY INDEXING:  a[[1,3,5]]    → Returns a COPY (independent)
BOOLEAN INDEXING: a[a > 5]     → Returns a COPY (independent)
```

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week01_numpy_foundations/day03_fancy_boolean_indexing.py`

**Exercise 1:** Create a 10×10 matrix. Use fancy indexing to extract rows 0, 3, 7.
```python
M = np.arange(100).reshape(10, 10)
selected_rows = M[[0, 3, 7]]
print(f"Shape: {selected_rows.shape}")  # Expected: (3, 10)
print(selected_rows)
```

**Exercise 2:** Create a 1D array of 100 random values from Normal(0, 1). Select all values between -1 and 1.
```python
rng = np.random.default_rng(42)
data = rng.standard_normal(100)

# Boolean mask: values between -1 and 1
mask = (data >= -1) & (data <= 1)  # Note: use & not 'and' for arrays
filtered = data[mask]

print(f"Total values: {len(data)}")
print(f"Values in [-1, 1]: {len(filtered)}")
print(f"Percentage: {len(filtered)/len(data)*100:.1f}%")
# Expected: ~68% (by the 68-95-99.7 rule for normal distribution)
```

**Exercise 3:** Use `np.ix_` to extract a submatrix.
```python
M = np.arange(25).reshape(5, 5)
# Extract rows [0, 2, 4] and columns [1, 3]
rows = [0, 2, 4]
cols = [1, 3]
submatrix = M[np.ix_(rows, cols)]
print(f"Shape: {submatrix.shape}")  # Expected: (3, 2)
print(f"Submatrix:\n{submatrix}")
```

**Exercise 4: IMPLEMENT ReLU** — this is your first neural network function!
```python
def relu_with_where(x):
    """ReLU activation: max(0, x) using np.where"""
    return np.where(x > 0, x, 0)

def relu_with_clip(x):
    """ReLU activation: max(0, x) using np.clip"""
    return np.clip(x, 0, None)

def relu_with_mask(x):
    """ReLU activation: max(0, x) using boolean masking"""
    result = x.copy()
    result[result < 0] = 0
    return result

# Test all three
test = np.array([-3, -1, 0, 1, 3, 5, -2, 4])
print(f"Input:          {test}")
print(f"relu (where):   {relu_with_where(test)}")
print(f"relu (clip):    {relu_with_clip(test)}")
print(f"relu (mask):    {relu_with_mask(test)}")
# All should give: [0, 0, 0, 1, 3, 5, 0, 4]

# Verify all give same result
assert np.array_equal(relu_with_where(test), relu_with_clip(test))
assert np.array_equal(relu_with_where(test), relu_with_mask(test))
print("\n✅ All three implementations match!")
```

**Exercise 5:** Count how many values are positive in a 100-element random array.
```python
rng = np.random.default_rng(42)
data = rng.standard_normal(100)

# Method 1: Boolean sum
count1 = np.sum(data > 0)
# Method 2: Count nonzero
count2 = np.count_nonzero(data > 0)

print(f"Positive values: {count1}")
print(f"Matches: {count1 == count2}")
# Expected: ~50 (half of normal distribution)
```

**Exercise 6:** Given student scores (10 students × 5 subjects), find students who scored >80 in ALL subjects.
```python
rng = np.random.default_rng(42)
scores = rng.integers(50, 100, size=(10, 5))
print("All scores:")
print(scores)

# Find students where EVERY subject > 80
all_above_80 = np.all(scores > 80, axis=1)  # axis=1 means "across columns"
top_students = np.where(all_above_80)[0]

print(f"\nStudents with ALL scores > 80: {top_students}")
print(f"Their scores:\n{scores[all_above_80]}")
```

### ✅ Day 3 Checklist
- [ ] I know the difference: slicing=VIEW, fancy/boolean indexing=COPY
- [ ] I can filter arrays using boolean conditions with `&`, `|`, `~`
- [ ] I implemented ReLU three different ways and they all match
- [ ] I understand `np.where(condition, true_val, false_val)`
- [ ] I understand `axis` parameter in `np.all()` and `np.any()`
- [ ] Code is pushed to GitHub

---

## 📅 Day 4 — Thursday, Aug 7: Broadcasting & Element-wise Operations

### 🌅 Morning (6:00 – 7:30 AM) — Read

**What to read:** VanderPlas Ch. 2.5 ("Computation on Arrays: Broadcasting")

**THE THREE RULES OF BROADCASTING (memorize these):**
```
Rule 1: If arrays differ in ndim, pad the SMALLER shape with 1s on the LEFT
Rule 2: Arrays with size 1 along a dimension act as if they had the size of
        the larger array along that dimension (they get "stretched")
Rule 3: If sizes disagree AND neither is 1 → ERROR (incompatible shapes)
```

**Practice predicting shapes WITHOUT running code (write on paper):**
```
(3, 4) + (4,)       →  (3, 4) + (1, 4) → (3, 4)     ✅ works
(3, 1) + (1, 4)     →  (3, 4)                          ✅ works
(3, 4) + (3,)       →  (3, 4) + (1, 3) → ERROR!       ❌ 4≠3
(2, 3, 4) + (3, 4)  →  (2, 3, 4) + (1, 3, 4) → (2, 3, 4)  ✅ works
(5, 1, 3) + (1, 4, 1) →  (5, 4, 3)             ✅ works
```

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week01_numpy_foundations/day04_broadcasting.py`

**Exercise 1:** Verify the 5 broadcasting predictions from above.
```python
# Test each prediction — does it match what you wrote on paper?
# For ones that work, print the result shape
# For ones that should fail, use try/except to catch the error

# Case 1: (3,4) + (4,)
a = np.ones((3, 4))
b = np.ones((4,))
result = a + b
print(f"(3,4) + (4,) → {result.shape}")  # Expected: (3, 4)

# Case 2: (3,1) + (1,4)
a = np.ones((3, 1))
b = np.ones((1, 4))
result = a + b
print(f"(3,1) + (1,4) → {result.shape}")  # Expected: (3, 4)

# Case 3: (3,4) + (3,) — THIS SHOULD FAIL
try:
    a = np.ones((3, 4))
    b = np.ones((3,))
    result = a + b
    print(f"(3,4) + (3,) → {result.shape}")
except ValueError as e:
    print(f"(3,4) + (3,) → ERROR: {e}")

# Case 4 and 5: do these yourself
```

**Exercise 2:** 10×10 multiplication table using broadcasting.
```python
row = np.arange(1, 11).reshape(10, 1)   # Shape: (10, 1) — column vector
col = np.arange(1, 11).reshape(1, 10)   # Shape: (1, 10) — row vector

table = row * col  # Broadcasting: (10, 1) * (1, 10) → (10, 10)
print("Multiplication Table:")
print(table)
# Verify: table[6, 7] should be 7*8 = 56
assert table[6, 7] == 56, "Something is wrong"
```

**Exercise 3:** Subtract channel means from RGB pixels (image normalization — used in every vision model).
```python
# Simulating a batch of 100 RGB pixels (values 0–255)
rng = np.random.default_rng(42)
pixels = rng.integers(0, 256, size=(100, 3)).astype(np.float64)

# ImageNet channel means (these are THE standard values used everywhere)
imagenet_means = np.array([0.485, 0.456, 0.406]) * 255  # Scale to 0-255 range

# Subtract means using broadcasting: (100, 3) - (3,) → (100, 3)
normalized = pixels - imagenet_means
print(f"Before - channel means: {pixels.mean(axis=0)}")
print(f"After  - channel means: {normalized.mean(axis=0)}")
# After normalization, means should be near 0
```

**Exercise 4:** Compute f(x) = x² + 3x + 2 and plot it.
```python
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1000)
y = x**2 + 3*x + 2  # All broadcasting — x is (1000,), scalars broadcast

plt.figure(figsize=(8, 5))
plt.plot(x, y, linewidth=2, color='#4A90D9')
plt.axhline(y=0, color='gray', linewidth=0.5)
plt.axvline(x=0, color='gray', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('f(x) = x² + 3x + 2')
plt.title('My First Mathematical Plot')
plt.grid(True, alpha=0.3)
plt.savefig('week01_numpy_foundations/day04_plot.png', dpi=150)
plt.show()
print("✅ Plot saved!")
```

**Exercise 5:** Outer product using broadcasting (no np.outer).
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6, 7])

# Reshape a to (3, 1) and b to (1, 4) → broadcasting gives (3, 4)
outer = a[:, np.newaxis] * b[np.newaxis, :]
print(f"Outer product (broadcasting):\n{outer}")

# Verify against np.outer
assert np.array_equal(outer, np.outer(a, b)), "Mismatch!"
print("✅ Matches np.outer!")
```

**Exercise 6:** Subtract row mean from each row (zero-centering — used in PCA, normalization).
```python
rng = np.random.default_rng(42)
X = rng.random((50, 10)) * 100  # 50 samples, 10 features

# Compute row means — shape should be (50, 1) for broadcasting
row_means = X.mean(axis=1, keepdims=True)  # keepdims=True is CRITICAL
print(f"Row means shape: {row_means.shape}")  # Must be (50, 1), NOT (50,)

# Subtract
X_centered = X - row_means  # (50, 10) - (50, 1) → (50, 10)

# Verify each row now has mean ≈ 0
print(f"Row means after centering (should be ~0): {X_centered.mean(axis=1)[:5]}")
assert np.allclose(X_centered.mean(axis=1), 0), "Centering failed!"
print("✅ All row means are ~0!")

# NOW TRY WITHOUT keepdims — see what breaks:
row_means_bad = X.mean(axis=1)  # Shape: (50,) — no second dimension!
print(f"\nWithout keepdims, shape: {row_means_bad.shape}")
try:
    X_bad = X - row_means_bad  # (50, 10) - (50,) → this broadcasts WRONG
    print(f"This 'worked' but gives wrong result!")
    print(f"Row means: {X_bad.mean(axis=1)[:5]}")  # NOT zero!
except Exception as e:
    print(f"Error: {e}")
```
> **Write in your notes:** "`keepdims=True` preserves the dimension so broadcasting works correctly. ALWAYS use it when subtracting means/stds along an axis."

### ✅ Day 4 Checklist
- [ ] I can predict broadcasting shapes on paper for any pair of arrays
- [ ] I know the 3 rules of broadcasting
- [ ] I know why `keepdims=True` matters and when to use it
- [ ] I normalized an "image" using channel means (this is real ML preprocessing)
- [ ] I plotted a mathematical function with matplotlib
- [ ] Code is pushed to GitHub

---

## 📅 Day 5 — Friday, Aug 8: Aggregation, Axis Operations & Sorting

### 🌅 Morning (6:00 – 7:30 AM) — Read

**What to read:** VanderPlas Ch. 2.4 ("Aggregations: Min, Max, and Everything In Between") + Ch. 2.8 ("Sorting Arrays")

**The `axis` parameter — the most confusing thing in NumPy:**
```
Given a matrix of shape (ROWS, COLUMNS):

np.sum(M, axis=0)  → sum DOWN each column → result shape: (COLUMNS,)
                      "collapse the rows"

np.sum(M, axis=1)  → sum ACROSS each row → result shape: (ROWS,)
                      "collapse the columns"

np.sum(M)           → sum everything → single number

MEMORY TRICK: axis=N means "collapse dimension N"
              axis=0 collapses rows    → answer per column
              axis=1 collapses columns → answer per row
```

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week01_numpy_foundations/day05_aggregation_axis.py`

**Exercise 1:** Create a (5, 4) matrix. Compute total sum, column sums, row sums. Verify consistency.
```python
M = np.arange(20).reshape(5, 4)
print(f"Matrix:\n{M}\n")

total = np.sum(M)
col_sums = np.sum(M, axis=0)  # Sum DOWN → one value per column
row_sums = np.sum(M, axis=1)  # Sum ACROSS → one value per row

print(f"Total sum: {total}")
print(f"Column sums (axis=0): {col_sums}")  # Shape: (4,)
print(f"Row sums (axis=1): {row_sums}")      # Shape: (5,)
print(f"Sum of col_sums: {col_sums.sum()}")  # Should equal total
print(f"Sum of row_sums: {row_sums.sum()}")  # Should equal total

assert total == col_sums.sum() == row_sums.sum(), "Math is broken"
print("✅ All sums are consistent!")
```

**Exercise 2: STANDARDIZATION — this is used in EVERY ML pipeline**
```python
rng = np.random.default_rng(42)
X = rng.random((100, 5)) * 100 + 50  # 100 samples, 5 features, values ~50-150

print("BEFORE standardization:")
print(f"  Column means: {X.mean(axis=0)}")
print(f"  Column stds:  {X.std(axis=0)}")

# Standardize: subtract mean, divide by std (per column)
col_mean = X.mean(axis=0, keepdims=True)  # Shape: (1, 5)
col_std = X.std(axis=0, keepdims=True)    # Shape: (1, 5)
X_std = (X - col_mean) / col_std

print("\nAFTER standardization:")
print(f"  Column means: {X_std.mean(axis=0)}")  # Should be ~0
print(f"  Column stds:  {X_std.std(axis=0)}")    # Should be ~1

assert np.allclose(X_std.mean(axis=0), 0, atol=1e-10), "Means not zero!"
assert np.allclose(X_std.std(axis=0), 1, atol=1e-10), "Stds not one!"
print("✅ Standardization correct!")
```

**Exercise 3:** Find the index of highest score in each row.
```python
rng = np.random.default_rng(42)
scores = rng.integers(0, 100, size=(20, 10))  # 20 students, 10 subjects

best_subject_per_student = np.argmax(scores, axis=1)  # Index of max in each row
best_score_per_student = np.max(scores, axis=1)

for i in range(5):  # Print first 5
    print(f"Student {i}: best subject = {best_subject_per_student[i]}, "
          f"score = {best_score_per_student[i]}")
```

**Exercise 4:** Find top 5 values efficiently using `np.argpartition` (faster than full sort).
```python
rng = np.random.default_rng(42)
data = rng.standard_normal(1000)

# np.argpartition is O(n) — much faster than full sort O(n log n)
k = 5
top_k_indices = np.argpartition(data, -k)[-k:]  # Indices of top 5
top_k_values = data[top_k_indices]

# Sort them for display
sorted_order = np.argsort(top_k_values)[::-1]
top_k_sorted = top_k_values[sorted_order]

print(f"Top {k} values: {top_k_sorted}")
print(f"  at indices: {top_k_indices[sorted_order]}")
```

**Exercise 5: L2 NORM — you will use this in EVERY distance calculation in ML**
```python
rng = np.random.default_rng(42)
X = rng.random((100, 3))  # 100 points in 3D space

# L2 norm of each row (each point's distance from origin)
# ||x||₂ = sqrt(x₁² + x₂² + x₃²)

# Method 1: Manual
norms_manual = np.sqrt(np.sum(X**2, axis=1))

# Method 2: np.linalg.norm
norms_builtin = np.linalg.norm(X, axis=1)

print(f"Manual norms (first 5):  {norms_manual[:5]}")
print(f"Built-in norms (first 5): {norms_builtin[:5]}")
assert np.allclose(norms_manual, norms_builtin), "Mismatch!"
print(f"✅ Both methods match! Shape: {norms_manual.shape}")  # (100,)
```

**Exercise 6: The `keepdims` demo (again, because it's THAT important)**
```python
M = np.array([[1, 2, 3],
              [4, 5, 6]])

# WITHOUT keepdims
means = M.mean(axis=1)
print(f"Without keepdims: shape={means.shape}, values={means}")
# Shape: (2,) — this is a 1D array

# WITH keepdims
means_k = M.mean(axis=1, keepdims=True)
print(f"With keepdims:    shape={means_k.shape}, values={means_k.ravel()}")
# Shape: (2, 1) — this is a 2D column vector

# Try subtracting:
try:
    result_bad = M - means  # (2, 3) - (2,) → broadcasts COLUMNS, not rows!
    print(f"\nWithout keepdims, M - means:\n{result_bad}")
    print(f"Row means of result: {result_bad.mean(axis=1)}")
    # These are NOT zero! Broadcasting went wrong!
except:
    pass

result_good = M - means_k  # (2, 3) - (2, 1) → correct broadcasting
print(f"\nWith keepdims, M - means:\n{result_good}")
print(f"Row means of result: {result_good.mean(axis=1)}")
# These ARE zero!
```

### ✅ Day 5 Checklist
- [ ] I can use `axis=0` and `axis=1` without confusion
- [ ] I remember the trick: "axis=N collapses dimension N"
- [ ] I standardized a dataset (subtract mean, divide by std per column)
- [ ] I computed L2 norms both manually and with `np.linalg.norm`
- [ ] I deeply understand why `keepdims=True` is necessary
- [ ] Code is pushed to GitHub

---

## 📅 Day 6 — Saturday, Aug 9: Stacking, Splitting & Reshaping

### 🌅 Morning (6:00 – 7:30 AM) — Read

**What to read:** VanderPlas Ch. 2.2 (the reshaping section) + Rougier Ch. 2 (memory layout)

**Key functions cheat sheet:**
```
RESHAPING:
  .reshape(new_shape)     — change shape, same data
  .ravel()                — flatten to 1D (returns view if possible)
  .flatten()              — flatten to 1D (always returns copy)
  .T                      — transpose (swap rows & columns)
  np.expand_dims(a, axis) — add a dimension
  np.squeeze(a)           — remove dimensions of size 1

COMBINING:
  np.concatenate([a, b], axis=0) — join along existing axis
  np.vstack([a, b])              — stack vertically (add rows)
  np.hstack([a, b])              — stack horizontally (add columns)
  np.stack([a, b], axis=0)       — stack along NEW axis

SPLITTING:
  np.split(a, n, axis=0)  — split into n equal parts
  np.array_split(a, n)    — split into n parts (allows unequal)
```

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week01_numpy_foundations/day06_reshape_stack.py`

**Exercise 1:** Stack 5 matrices into a batch (this is how image batches work).
```python
# In deep learning, images come in batches: (batch_size, height, width)
images = [np.random.rand(28, 28) for _ in range(5)]

batch = np.stack(images, axis=0)
print(f"Batch shape: {batch.shape}")  # Expected: (5, 28, 28)
# This is exactly how MNIST data is shaped!
```

**Exercise 2:** Reshape flattened MNIST back to images.
```python
# MNIST images are 28×28 = 784 pixels
# They're often stored flattened as (num_images, 784)
rng = np.random.default_rng(42)
flattened = rng.random((100, 784))  # 100 fake "images"

# Reshape each 784-vector back to 28×28
images = flattened.reshape(100, 28, 28)
print(f"Flattened shape: {flattened.shape}")  # (100, 784)
print(f"Images shape:    {images.shape}")     # (100, 28, 28)

# Verify: first image's first pixel should match
assert flattened[0, 0] == images[0, 0, 0], "Reshape broke the data!"
print("✅ Reshape preserved the data!")
```

**Exercise 3:** Train/test split using array splitting.
```python
rng = np.random.default_rng(42)
X = rng.random((1000, 10))  # 1000 samples, 10 features
y = rng.integers(0, 2, 1000)  # Binary labels

# Shuffle first (important! data might be ordered by class)
indices = rng.permutation(1000)
X_shuffled = X[indices]
y_shuffled = y[indices]

# 80/20 split
split_point = int(0.8 * 1000)  # = 800
X_train, X_test = X_shuffled[:split_point], X_shuffled[split_point:]
y_train, y_test = y_shuffled[:split_point], y_shuffled[split_point:]

print(f"Train: X={X_train.shape}, y={y_train.shape}")  # (800, 10), (800,)
print(f"Test:  X={X_test.shape},  y={y_test.shape}")    # (200, 10), (200,)
print(f"Train class balance: {np.bincount(y_train)}")
print(f"Test class balance:  {np.bincount(y_test)}")
```

**Exercise 4:** vstack and hstack.
```python
a = np.ones((50, 3))
b = np.zeros((50, 3))

vertical = np.vstack([a, b])
print(f"vstack: {vertical.shape}")  # (100, 3)

horizontal = np.hstack([a, b])
print(f"hstack: {horizontal.shape}")  # (50, 6)
```

**Exercise 5:** Transpose a (batch, seq_len, hidden) tensor — used in Transformers.
```python
# In Transformers, you often need to swap seq_len and hidden dimensions
T = np.random.rand(2, 10, 64)  # batch=2, seq_len=10, hidden=64
print(f"Original: {T.shape}")   # (2, 10, 64)

# Swap last two dimensions
T_transposed = T.transpose(0, 2, 1)  # Keep batch, swap seq and hidden
print(f"Transposed: {T_transposed.shape}")  # (2, 64, 10)

# Alternative: np.swapaxes
T_swapped = np.swapaxes(T, 1, 2)
print(f"Swapped:    {T_swapped.shape}")  # (2, 64, 10)

assert np.array_equal(T_transposed, T_swapped)
print("✅ Both methods give same result!")
```

**Exercise 6:** Create a mini-batch generator (used in every training loop).
```python
def get_batches(X, y, batch_size, rng):
    """Yield mini-batches of (X_batch, y_batch)"""
    n_samples = X.shape[0]
    indices = rng.permutation(n_samples)  # Shuffle each epoch
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]

# Test it
rng = np.random.default_rng(42)
X = np.arange(100).reshape(50, 2)  # 50 samples, 2 features
y = np.arange(50)

print("Batches:")
for i, (X_batch, y_batch) in enumerate(get_batches(X, y, batch_size=16, rng=rng)):
    print(f"  Batch {i}: X shape={X_batch.shape}, y shape={y_batch.shape}")
# Expected: 3 batches of 16 + 1 batch of 2
```

### ✅ Day 6 Checklist
- [ ] I can reshape any array to any compatible shape
- [ ] I understand the difference between `.ravel()` (view) and `.flatten()` (copy)
- [ ] I can stack arrays into batches using `np.stack`
- [ ] I can do train/test splitting manually
- [ ] I built a mini-batch generator (used in ML training loops)
- [ ] I can transpose multi-dimensional tensors
- [ ] Code is pushed to GitHub

---

## 🔄 Sunday, Aug 10 — REST DAY
> Do not study. Rest. Exercise. See friends. Your brain consolidates learning during rest.
> If you feel the urge to do something: review your handwritten notes for 20 minutes maximum.

---

# WEEK 2 — LINEAR ALGEBRA PART 1 (STRANG LECTURES 1-10)
### Aug 11 (Mon) – Aug 16 (Sat)
> *Now the math begins. Every morning: a Strang lecture. Every evening: implement it in NumPy.*
> **Source:** [MIT 18.06 — Gilbert Strang](https://www.youtube.com/playlist?list=PLE7DDD91010BC51F8)
> Lectures are ~50 min each. Watch at 1.25x speed = ~40 min. Take notes the remaining 50 min.

---

## 📅 Day 7 — Monday, Aug 11: Week 1 Review + Geometry of Linear Equations

### 🌅 Morning (6:00 – 7:30 AM) — Week 1 Review + Math

**First 30 minutes:** Do these 5 NumPy problems WITHOUT looking at your previous code:
1. Create a (3, 4, 5) tensor. Compute the mean along axis=2 with keepdims=True. State the output shape.
2. Given array `a = np.array([-3, 1, -1, 4, -2, 5])`, apply ReLU using `np.where`.
3. Create a (6, 6) matrix. Extract the 3×3 center block.
4. Given two arrays of shape (50, 3) and (1, 3), predict the output shape of their sum.
5. Flatten a (10, 28, 28) tensor to (10, 784).

**If you got all 5 right without looking anything up → Week 1 is complete. Move on.**
**If not → redo the relevant Day's exercises before proceeding.**

**Remaining 60 minutes:** Watch & take notes:
- **Video:** [MIT 18.06 Lecture 1 — The Geometry of Linear Equations](https://www.youtube.com/watch?v=J7DzL2_Na80) (39 min)
- **Key concepts:** Row picture vs Column picture of Ax = b. Linear combinations.

**Notes to write (by hand):**
- Draw the row picture for a 2×2 system: two lines intersecting
- Draw the column picture: combining two column vectors to reach b
- Write: "Ax is a LINEAR COMBINATION of the columns of A"

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week02_linear_algebra_part1/day07_geometry_of_equations.py`

**Exercise 1:** Solve Ax = b and verify.
```python
A = np.array([[2, -1],
              [-1, 2]], dtype=float)
b = np.array([0, 3], dtype=float)

# Solve using NumPy
x = np.linalg.solve(A, b)
print(f"Solution x: {x}")

# Verify: A @ x should equal b
residual = A @ x - b
print(f"Residual (should be ~0): {residual}")
assert np.allclose(A @ x, b), "Solution is wrong!"
print("✅ Ax = b verified!")
```

**Exercise 2:** Plot the row picture (two lines intersecting).
```python
import matplotlib.pyplot as plt

# System: 2x - y = 0  and  -x + 2y = 3
x_vals = np.linspace(-1, 4, 100)

# Row 1: 2x - y = 0 → y = 2x
y1 = 2 * x_vals

# Row 2: -x + 2y = 3 → y = (x + 3) / 2
y2 = (x_vals + 3) / 2

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y1, label='2x - y = 0', linewidth=2)
plt.plot(x_vals, y2, label='-x + 2y = 3', linewidth=2)

# Mark the intersection (the solution)
x_sol = np.linalg.solve(A, b)
plt.plot(x_sol[0], x_sol[1], 'ro', markersize=10, label=f'Solution ({x_sol[0]:.1f}, {x_sol[1]:.1f})')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Row Picture: Two Lines Intersecting')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.savefig('week02_linear_algebra_part1/day07_row_picture.png', dpi=150)
plt.show()
```

**Exercise 3:** Plot the column picture (combining column vectors).
```python
plt.figure(figsize=(8, 6))

# Columns of A
col1 = A[:, 0]  # [2, -1]
col2 = A[:, 1]  # [-1, 2]

# Solution: x1 * col1 + x2 * col2 = b
origin = [0, 0]

# Draw x1 * col1
plt.arrow(0, 0, x_sol[0]*col1[0], x_sol[0]*col1[1],
          head_width=0.1, color='blue', label=f'{x_sol[0]:.1f} × col1')

# Draw x2 * col2 starting from end of x1*col1
start = x_sol[0] * col1
plt.arrow(start[0], start[1], x_sol[1]*col2[0], x_sol[1]*col2[1],
          head_width=0.1, color='red', label=f'{x_sol[1]:.1f} × col2')

# Mark b
plt.plot(b[0], b[1], 'go', markersize=10, label=f'b = ({b[0]}, {b[1]})')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Column Picture: Linear Combination of Columns')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.savefig('week02_linear_algebra_part1/day07_column_picture.png', dpi=150)
plt.show()
```

**Exercise 4:** Solve a 3×3 system. Change b and solve again.
```python
A3 = np.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 10]], dtype=float)  # Note: [7,8,10] not [7,8,9] (that would be singular)

b3 = np.array([1, 2, 3], dtype=float)
x3 = np.linalg.solve(A3, b3)
print(f"Solution for b=[1,2,3]: {x3}")
assert np.allclose(A3 @ x3, b3)

# Change b
b3_new = np.array([0, 0, 1], dtype=float)
x3_new = np.linalg.solve(A3, b3_new)
print(f"Solution for b=[0,0,1]: {x3_new}")
assert np.allclose(A3 @ x3_new, b3_new)
print("✅ Both solutions verified!")
```

**Exercise 5:** Try a system with no solution (catch the error).
```python
# Singular matrix — rows are linearly dependent
A_bad = np.array([[1, 2],
                   [2, 4]], dtype=float)  # Row 2 = 2 × Row 1
b_bad = np.array([1, 3], dtype=float)      # 3 ≠ 2*1, so no solution

try:
    x_bad = np.linalg.solve(A_bad, b_bad)
    print(f"Solution: {x_bad}")  # This shouldn't happen
except np.linalg.LinAlgError as e:
    print(f"✅ Caught expected error: {e}")
    print("   This matrix is singular — the system has no unique solution")
```

### ✅ Day 7 Checklist
- [ ] I passed the Week 1 review (all 5 problems without looking)
- [ ] I watched Strang Lecture 1 and took handwritten notes
- [ ] I can explain the difference between row picture and column picture
- [ ] I solved Ax = b in NumPy and plotted both pictures
- [ ] I handled a singular matrix gracefully with try/except
- [ ] Code + plots are pushed to GitHub

---

## 📅 Day 8 — Tuesday, Aug 12: Elimination & Matrices

### 🌅 Morning — Watch [Strang Lecture 2](https://www.youtube.com/watch?v=QVKj3LADCnA) (52 min at 1.25x ≈ 42 min)
**Key concepts:** Gaussian elimination. Elimination matrices. Back substitution. Pivot positions.

**Notes to write:**
- The elimination process step by step for a 3×3 matrix
- What a "pivot" is and what happens when a pivot is zero
- What an elimination matrix E looks like (identity matrix with one off-diagonal changed)

### 🌙 Evening — Code

**File:** `week02_linear_algebra_part1/day08_gaussian_elimination.py`

**THE BIG EXERCISE: Implement Gaussian Elimination from scratch.**
```python
def gaussian_elimination(A, b):
    """
    Solve Ax = b using Gaussian elimination + back substitution.
    
    Steps:
    1. Forward elimination: convert A to upper triangular form
    2. Back substitution: solve from bottom row up
    
    Parameters:
        A: numpy array of shape (n, n)
        b: numpy array of shape (n,)
    
    Returns:
        x: numpy array of shape (n,) — the solution
    """
    n = A.shape[0]
    
    # Create augmented matrix [A | b]
    # We work on a copy so we don't modify the originals
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])
    
    # === FORWARD ELIMINATION ===
    for col in range(n):
        # Find the pivot (the element Ab[col, col])
        # If it's zero, we need to swap rows (partial pivoting)
        if abs(Ab[col, col]) < 1e-12:
            # Find a row below with a non-zero entry in this column
            for swap_row in range(col + 1, n):
                if abs(Ab[swap_row, col]) > 1e-12:
                    Ab[[col, swap_row]] = Ab[[swap_row, col]]  # Swap rows
                    break
            else:
                raise ValueError("Matrix is singular — no unique solution")
        
        # Eliminate all entries below the pivot
        for row in range(col + 1, n):
            factor = Ab[row, col] / Ab[col, col]
            Ab[row] = Ab[row] - factor * Ab[col]
    
    # === BACK SUBSTITUTION ===
    x = np.zeros(n)
    for row in range(n - 1, -1, -1):  # Start from last row, go up
        x[row] = (Ab[row, -1] - Ab[row, row+1:n] @ x[row+1:n]) / Ab[row, row]
    
    return x

# TEST IT
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)

x_mine = gaussian_elimination(A, b)
x_numpy = np.linalg.solve(A, b)

print(f"My solution:    {x_mine}")
print(f"NumPy solution: {x_numpy}")
print(f"Match: {np.allclose(x_mine, x_numpy)}")

# Verify
print(f"A @ x = {A @ x_mine}")
print(f"b     = {b}")
assert np.allclose(A @ x_mine, b), "Solution is wrong!"
print("✅ Gaussian elimination works!")
```

**Exercise 2:** Construct the elimination matrix E₂₁ that subtracts 3× row 1 from row 2.
```python
# E₂₁ is an identity matrix with E[1, 0] = -3
E21 = np.eye(3)
E21[1, 0] = -3

A = np.array([[1, 2, 1],
              [3, 8, 1],
              [0, 4, 1]], dtype=float)

result = E21 @ A
print(f"E21:\n{E21}")
print(f"\nA:\n{A}")
print(f"\nE21 @ A:\n{result}")
# Row 2 of result should be: [3,8,1] - 3*[1,2,1] = [0, 2, -2]
assert np.allclose(result[1], [0, 2, -2]), "Elimination matrix is wrong!"
print("✅ Elimination matrix works!")
```

**Exercise 3:** Test your Gaussian elimination on a 5×5 system.
```python
rng = np.random.default_rng(42)
A5 = rng.random((5, 5)) * 10
b5 = rng.random(5) * 10

x5_mine = gaussian_elimination(A5, b5)
x5_numpy = np.linalg.solve(A5, b5)

print(f"My solution:    {x5_mine}")
print(f"NumPy solution: {x5_numpy}")
assert np.allclose(x5_mine, x5_numpy, atol=1e-8), "5x5 solution doesn't match!"
print("✅ Works on 5×5 system!")
```

### ✅ Day 8 Checklist
- [ ] Watched Strang Lecture 2 and took handwritten notes
- [ ] Implemented Gaussian elimination from scratch (forward + back sub)
- [ ] My implementation gives the same answer as `np.linalg.solve`
- [ ] I understand what happens when a pivot is zero (row swap)
- [ ] I built an elimination matrix and verified E @ A gives the right result
- [ ] Tested on a 5×5 system and it works
- [ ] Code is pushed to GitHub

---

## 📅 Day 9 — Wednesday, Aug 13: Matrix Operations & Inverses

### 🌅 Morning — Watch [Strang Lecture 3](https://www.youtube.com/watch?v=FX4C-JpTFgY) (51 min at 1.25x ≈ 41 min)
**Key concepts:** Matrix multiplication (4 ways to think about it). Why row×column.

Also watch (if time permits): [Strang Lecture 4](https://www.youtube.com/watch?v=5hO3MrzPa0A) — A = LU (LU decomposition)

**Notes to write:**
- The 4 ways to see matrix multiplication: (1) dot products, (2) columns of C, (3) rows of C, (4) sum of outer products
- What A⁻¹ means: the matrix that "undoes" A
- What LU decomposition is: A = Lower × Upper triangular

### 🌙 Evening — Code

**File:** `week02_linear_algebra_part1/day09_matrix_multiply_inverse.py`

**Exercise 1:** Implement matrix multiply with 3 nested loops. Time it vs `@`.
```python
import time

def my_matmul(A, B):
    """Matrix multiply using 3 nested loops — SLOW but educational"""
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, f"Shape mismatch: {A.shape} and {B.shape}"
    
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

# Test correctness
A = np.array([[1, 2], [3, 4]], dtype=float)
B = np.array([[5, 6], [7, 8]], dtype=float)
C_mine = my_matmul(A, B)
C_numpy = A @ B
assert np.allclose(C_mine, C_numpy), "My matmul is wrong!"
print(f"✅ 2x2 correct: {C_mine}")

# Time comparison on larger matrices
n = 100
A_big = np.random.rand(n, n)
B_big = np.random.rand(n, n)

start = time.time()
C_loop = my_matmul(A_big, B_big)
time_loop = time.time() - start

start = time.time()
C_numpy = A_big @ B_big
time_numpy = time.time() - start

print(f"\n100×100 matrix multiply:")
print(f"  My loops:  {time_loop:.4f} seconds")
print(f"  NumPy @:   {time_numpy:.6f} seconds")
print(f"  Speedup:   {time_loop/time_numpy:.0f}x")
# NumPy will be 100-1000x faster. This is why we use NumPy!
```

**Exercise 2:** Compute A⁻¹ and verify A @ A⁻¹ ≈ I.
```python
A = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]], dtype=float)

A_inv = np.linalg.inv(A)
product = A @ A_inv
identity = np.eye(3)

print(f"A:\n{A}")
print(f"\nA⁻¹:\n{A_inv}")
print(f"\nA @ A⁻¹:\n{product}")
print(f"\nIs A @ A⁻¹ ≈ I? {np.allclose(product, identity)}")
assert np.allclose(product, identity), "Inverse is wrong!"
print("✅ Inverse verified!")
```

**Exercise 3:** Implement finding the inverse via Gauss-Jordan elimination.
```python
def find_inverse(A):
    """
    Find A⁻¹ by augmenting [A | I] and reducing to [I | A⁻¹]
    Uses our Gaussian elimination approach.
    """
    n = A.shape[0]
    # Augmented matrix: [A | I]
    AI = np.hstack([A.astype(float), np.eye(n)])
    
    # Forward elimination (make lower triangle = 0)
    for col in range(n):
        # Partial pivoting
        max_row = col + np.argmax(np.abs(AI[col:, col]))
        AI[[col, max_row]] = AI[[max_row, col]]
        
        if abs(AI[col, col]) < 1e-12:
            raise ValueError("Matrix is singular!")
        
        # Scale pivot row to make pivot = 1
        AI[col] = AI[col] / AI[col, col]
        
        # Eliminate all OTHER rows (not just below — this is Gauss-JORDAN)
        for row in range(n):
            if row != col:
                factor = AI[row, col]
                AI[row] = AI[row] - factor * AI[col]
    
    # The right half is now A⁻¹
    return AI[:, n:]

# Test
A = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]], dtype=float)

inv_mine = find_inverse(A)
inv_numpy = np.linalg.inv(A)

print(f"My inverse:\n{inv_mine}")
print(f"\nNumPy inverse:\n{inv_numpy}")
assert np.allclose(inv_mine, inv_numpy, atol=1e-10), "My inverse is wrong!"
print("\n✅ My Gauss-Jordan inverse matches NumPy!")
```

**Exercise 4:** Verify (AB)⁻¹ = B⁻¹A⁻¹ numerically.
```python
rng = np.random.default_rng(42)
A = rng.random((4, 4))
B = rng.random((4, 4))

# Method 1: (AB)⁻¹
AB_inv = np.linalg.inv(A @ B)

# Method 2: B⁻¹ A⁻¹
B_inv_A_inv = np.linalg.inv(B) @ np.linalg.inv(A)

print(f"(AB)⁻¹ ≈ B⁻¹A⁻¹ ? {np.allclose(AB_inv, B_inv_A_inv)}")
assert np.allclose(AB_inv, B_inv_A_inv)
print("✅ Property verified!")
```

### ✅ Day 9 Checklist
- [ ] Watched Strang Lectures 3 (and ideally 4)
- [ ] Implemented matrix multiplication with loops, saw the massive speed difference vs `@`
- [ ] Computed matrix inverse with `np.linalg.inv` and verified A @ A⁻¹ = I
- [ ] Implemented Gauss-Jordan elimination to find the inverse from scratch
- [ ] Verified (AB)⁻¹ = B⁻¹A⁻¹
- [ ] Code is pushed to GitHub

---

## 📅 Day 10 — Thursday, Aug 14: Transposes, Permutations & Vector Spaces

### 🌅 Morning — Watch [Strang Lecture 5](https://www.youtube.com/watch?v=JibVXBElKL0) (48 min at 1.25x ≈ 38 min)
**Key concepts:** Transposes. Symmetric matrices. Permutation matrices. Intro to vector spaces.

**Notes:** (Aᵀ)ᵢⱼ = Aⱼᵢ. (AB)ᵀ = BᵀAᵀ (reverse order!). A symmetric matrix satisfies A = Aᵀ.

### 🌙 Evening — Code

**File:** `week02_linear_algebra_part1/day10_transpose_symmetry.py`

**Exercise 1:** Verify (AB)ᵀ = BᵀAᵀ for random matrices.
```python
rng = np.random.default_rng(42)
A = rng.random((3, 4))
B = rng.random((4, 5))

lhs = (A @ B).T          # (AB)ᵀ — shape (5, 3)
rhs = B.T @ A.T           # BᵀAᵀ — shape (5, 3)

assert np.allclose(lhs, rhs), "Property violated!"
print(f"(AB)ᵀ shape: {lhs.shape}")
print(f"BᵀAᵀ shape:  {rhs.shape}")
print("✅ (AB)ᵀ = BᵀAᵀ verified!")
```

**Exercise 2:** Create a symmetric matrix using A @ A.T (always symmetric).
```python
rng = np.random.default_rng(42)
A = rng.random((4, 3))

S = A @ A.T  # This is ALWAYS symmetric, regardless of what A is
print(f"S = A @ Aᵀ:\n{S}")
print(f"\nIs S symmetric? {np.allclose(S, S.T)}")
assert np.allclose(S, S.T)
print("✅ AᵀA is always symmetric!")
```

**Exercise 3:** Prove AᵀA is always symmetric (even for non-square A).
```python
rng = np.random.default_rng(42)
# Try with various shapes
for shape in [(3, 5), (7, 2), (10, 10), (1, 8)]:
    A = rng.random(shape)
    AtA = A.T @ A
    assert np.allclose(AtA, AtA.T), f"Failed for shape {shape}!"
    print(f"  Shape {shape}: AᵀA shape = {AtA.shape}, symmetric = ✅")
print("✅ AᵀA is ALWAYS symmetric, for any shape!")
```

**Exercise 4:** Transpose a (batch, seq, features) tensor — used in Transformers.
```python
T = np.random.rand(2, 10, 64)
print(f"Original shape: {T.shape}")

T_swapped = T.transpose(0, 2, 1)
print(f"After transpose(0,2,1): {T_swapped.shape}")  # (2, 64, 10)

# Verify: T[0, 3, 7] should equal T_swapped[0, 7, 3]
assert T[0, 3, 7] == T_swapped[0, 7, 3]
print("✅ Values correctly transposed!")
```

### ✅ Day 10 Checklist
- [ ] Watched Strang Lecture 5
- [ ] Verified (AB)ᵀ = BᵀAᵀ
- [ ] Understand why AᵀA is always symmetric
- [ ] Can transpose multi-dimensional tensors for Transformer-style reshaping

---

## 📅 Day 11 — Friday, Aug 15: Column Space & Null Space

### 🌅 Morning — Watch [Strang Lectures 6-7](https://www.youtube.com/watch?v=8o5Cmfpeo6g) (watch at 1.5x, focus on key ideas)
**Key concepts:** Column space C(A), Null space N(A). Solving Ax = 0. Finding special solutions.

### 🌙 Evening — Code

**File:** `week02_linear_algebra_part1/day11_column_null_space.py`

```python
def analyze_matrix(A, name="A"):
    """Complete analysis of a matrix's fundamental spaces"""
    print(f"\n{'='*50}")
    print(f"Analysis of {name}:")
    print(f"{'='*50}")
    print(f"Shape: {A.shape}")
    print(f"Rank: {np.linalg.matrix_rank(A)}")
    
    U, S, Vt = np.linalg.svd(A)
    rank = np.sum(S > 1e-10)
    
    print(f"Singular values: {S}")
    print(f"Rank (from SVD): {rank}")
    
    # Null space = columns of V corresponding to zero singular values
    null_space = Vt[rank:].T
    if null_space.size > 0:
        print(f"Null space dimension: {null_space.shape[1]}")
        # Verify: A @ null_vector should ≈ 0
        for i in range(null_space.shape[1]):
            residual = np.linalg.norm(A @ null_space[:, i])
            print(f"  ||A @ null_vec_{i}|| = {residual:.2e}")
    else:
        print(f"Null space: trivial (only zero vector)")
    
    return rank, null_space

# Test with a rank-2 matrix (3x4, so it has a 2-dimensional null space)
A = np.array([[1, 2, 3, 4],
              [2, 4, 6, 8],   # Row 2 = 2 × Row 1
              [1, 3, 5, 7]])  # Independent from Row 1

rank, null = analyze_matrix(A, "Rank-deficient matrix")
print(f"\nRank + Nullity = {rank} + {null.shape[1]} = {rank + null.shape[1]}")
print(f"Number of columns = {A.shape[1]}")
print(f"Rank-Nullity theorem: {rank + null.shape[1]} = {A.shape[1]} ✅")
```

### ✅ Day 11 Checklist
- [ ] Watched Strang Lectures 6-7
- [ ] Understand column space = all possible Ax outputs
- [ ] Understand null space = all x where Ax = 0
- [ ] Can compute rank and null space using SVD
- [ ] Verified the Rank-Nullity theorem: rank + nullity = n

---

## 📅 Day 12 — Saturday, Aug 16: Orthogonality & Projections + Linear Algebra Review

### 🌅 Morning — Watch [Strang Lecture 14-15](https://www.youtube.com/watch?v=Y_Ac6KiQ1t0) (pick key segments about projections, ~45 min total)
**Key concepts:** Orthogonal vectors. Projection formula. Projection matrix. Gram-Schmidt process.

### 🌙 Evening — Code

**File:** `week02_linear_algebra_part1/day12_projections_gramschmidt.py`

**Exercise 1:** Project vector b onto vector a.
```python
def project_onto(b, a):
    """Project vector b onto vector a"""
    # p = (aᵀb / aᵀa) * a
    scalar = np.dot(a, b) / np.dot(a, a)
    projection = scalar * a
    return projection

a = np.array([1, 1, 1], dtype=float)
b = np.array([1, 2, 3], dtype=float)

p = project_onto(b, a)
error = b - p

print(f"a = {a}")
print(f"b = {b}")
print(f"projection = {p}")
print(f"error = {error}")
print(f"error · a = {np.dot(error, a):.10f}")  # Should be ~0 (orthogonal!)
assert abs(np.dot(error, a)) < 1e-10, "Error is not orthogonal to a!"
print("✅ Error is orthogonal to a!")
```

**Exercise 2:** Implement Gram-Schmidt orthogonalization from scratch.
```python
def gram_schmidt(V):
    """
    Gram-Schmidt process: take columns of V, make them orthonormal.
    
    Input:  V — matrix whose columns are the vectors to orthogonalize
    Output: Q — matrix whose columns are orthonormal
    """
    n, k = V.shape
    Q = np.zeros_like(V, dtype=float)
    
    for j in range(k):
        v = V[:, j].copy()
        
        # Subtract projections onto all previous orthonormal vectors
        for i in range(j):
            v = v - np.dot(Q[:, i], v) * Q[:, i]
        
        # Normalize
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            raise ValueError("Vectors are linearly dependent!")
        Q[:, j] = v / norm
    
    return Q

# Test with 3 non-orthogonal vectors
V = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]], dtype=float)

Q = gram_schmidt(V)
print(f"Original V:\n{V}")
print(f"\nOrthonormal Q:\n{Q}")

# Verify: QᵀQ should be identity
QtQ = Q.T @ Q
print(f"\nQᵀQ (should be I):\n{np.round(QtQ, 10)}")
assert np.allclose(QtQ, np.eye(3)), "Vectors are not orthonormal!"
print("✅ Gram-Schmidt produces orthonormal vectors!")

# Compare with NumPy's QR decomposition
Q_numpy, R_numpy = np.linalg.qr(V)
print(f"\nNumPy QR gives Q:\n{Q_numpy}")
# Note: signs may differ, but |Q| should match |Q_numpy|
print(f"Absolute values match: {np.allclose(np.abs(Q), np.abs(Q_numpy))}")
```

### ✅ Day 12 Checklist
- [ ] Understand projection formula: p = (aᵀb/aᵀa) × a
- [ ] Verified that the error (b - p) is orthogonal to a
- [ ] Implemented Gram-Schmidt from scratch
- [ ] Verified QᵀQ = I (orthonormal vectors)
- [ ] Compared with `np.linalg.qr()` result

---

## 🔄 Sunday, Aug 17 — REST DAY

---

# WEEK 3 — LINEAR ALGEBRA PART 2 (EIGENVALUES, SVD, PCA)
### Aug 18 (Mon) – Aug 23 (Sat)

> This week contains the math that **directly powers machine learning**: eigenvalues (PCA, stability), SVD (compression, dimensionality reduction), and norms (regularization, distances).

---

## 📅 Day 13 — Monday, Aug 18: Determinants

### 🌅 Morning — Watch [Strang Lecture 18](https://www.youtube.com/watch?v=srxexLishgY) (~50 min at 1.25x)
**Key concepts:** Properties of determinants. Cofactor expansion. det(AB) = det(A)×det(B). det = 0 means singular.

### 🌙 Evening — Code

**File:** `week03_linear_algebra_part1_contd/day13_determinants.py`

```python
# Exercise 1: Compute determinants with np.linalg.det
for n in [2, 3, 4, 5]:
    rng = np.random.default_rng(42)
    A = rng.random((n, n))
    print(f"{n}×{n} det = {np.linalg.det(A):.6f}")

# Exercise 2: Verify det(AB) = det(A) * det(B)
A = np.random.rand(4, 4)
B = np.random.rand(4, 4)
det_AB = np.linalg.det(A @ B)
det_A_times_det_B = np.linalg.det(A) * np.linalg.det(B)
print(f"\ndet(AB) = {det_AB:.6f}")
print(f"det(A)×det(B) = {det_A_times_det_B:.6f}")
assert np.allclose(det_AB, det_A_times_det_B)
print("✅ det(AB) = det(A)×det(B)")

# Exercise 3: Verify det(Aᵀ) = det(A)
det_A = np.linalg.det(A)
det_At = np.linalg.det(A.T)
assert np.allclose(det_A, det_At)
print(f"✅ det(A) = det(Aᵀ) = {det_A:.6f}")

# Exercise 4: Implement 2×2 and 3×3 determinant by hand
def det_2x2(M):
    return M[0,0]*M[1,1] - M[0,1]*M[1,0]

def det_3x3(M):
    """Cofactor expansion along first row"""
    return (M[0,0] * det_2x2(M[1:, [1,2]]) -
            M[0,1] * det_2x2(M[1:, [0,2]]) +
            M[0,2] * det_2x2(M[1:, [0,1]]))

M3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=float)
print(f"\nMy 3×3 det: {det_3x3(M3):.6f}")
print(f"NumPy det:  {np.linalg.det(M3):.6f}")
assert np.allclose(det_3x3(M3), np.linalg.det(M3))
print("✅ My determinant matches NumPy!")

# Exercise 5: Singular matrix — det = 0, inverse fails
S = np.array([[1, 2], [2, 4]], dtype=float)  # Row 2 = 2×Row 1
print(f"\nSingular matrix det: {np.linalg.det(S):.6f}")  # Should be ~0
```

### ✅ Day 13 Checklist
- [ ] Watched Strang Lecture 18
- [ ] Verified det(AB) = det(A)×det(B) and det(Aᵀ) = det(A)
- [ ] Implemented 2×2 and 3×3 determinant from scratch
- [ ] Understand: det = 0 ↔ singular ↔ no inverse

---

## 📅 Day 14 — Tuesday, Aug 19: Eigenvalues & Eigenvectors

### 🌅 Morning — Watch [Strang Lecture 21](https://www.youtube.com/watch?v=cdZnhQjJu4I) (~50 min)
Also highly recommended: [3Blue1Brown — Eigenvectors and Eigenvalues](https://www.youtube.com/watch?v=PFDu9oVAE-g) (15 min) — watch this FIRST for visual intuition.

**The core idea:** Av = λv. When you multiply A by v, the vector v doesn't change direction — it only gets scaled by λ. That's what "eigen" means.

### 🌙 Evening — Code

**File:** `week04_linear_algebra_part2/day14_eigenvalues.py`

```python
# Exercise 1: Compute eigenvalues and eigenvectors. Verify Av = λv.
A = np.array([[4, -2],
              [1,  1]], dtype=float)

eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues:  {eigenvalues}")
print(f"Eigenvectors (as columns):\n{eigenvectors}")

# Verify Av = λv for each eigenpair
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    lhs = A @ v        # Av
    rhs = lam * v       # λv
    print(f"\nEigenpair {i}: λ={lam:.4f}")
    print(f"  Av  = {lhs}")
    print(f"  λv  = {rhs}")
    assert np.allclose(lhs, rhs), f"Eigenpair {i} doesn't satisfy Av = λv!"
print("\n✅ All eigenpairs satisfy Av = λv!")

# Exercise 2: Verify trace = sum of eigenvalues, det = product of eigenvalues
print(f"\ntrace(A) = {np.trace(A):.4f}")
print(f"sum(eigenvalues) = {np.sum(eigenvalues):.4f}")
assert np.allclose(np.trace(A), np.sum(eigenvalues))

print(f"det(A) = {np.linalg.det(A):.4f}")
print(f"product(eigenvalues) = {np.prod(eigenvalues):.4f}")
assert np.allclose(np.linalg.det(A), np.prod(eigenvalues))
print("✅ trace = Σλ, det = Πλ")

# Exercise 3: Power iteration — find dominant eigenvalue from scratch
def power_iteration(A, num_iters=100, seed=42):
    """Find the largest eigenvalue and its eigenvector"""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[0])
    v = v / np.linalg.norm(v)
    
    for _ in range(num_iters):
        Av = A @ v
        v = Av / np.linalg.norm(Av)
    
    eigenvalue = v @ A @ v  # Rayleigh quotient
    return eigenvalue, v

lam_power, v_power = power_iteration(A)
print(f"\nPower iteration: λ = {lam_power:.6f}")
print(f"NumPy largest:   λ = {max(eigenvalues):.6f}")
print(f"Match: {np.allclose(lam_power, max(eigenvalues), atol=1e-4)}")

# Exercise 4: For symmetric matrix — eigenvalues are real, eigenvectors orthogonal
S = np.array([[2, 1], [1, 3]], dtype=float)  # Symmetric
vals, vecs = np.linalg.eigh(S)  # Use eigh for symmetric (faster, always real)
print(f"\nSymmetric eigenvalues (real): {vals}")
print(f"Eigenvectors orthogonal? dot = {np.dot(vecs[:, 0], vecs[:, 1]):.10f}")
assert abs(np.dot(vecs[:, 0], vecs[:, 1])) < 1e-10
print("✅ Symmetric matrix: real eigenvalues, orthogonal eigenvectors!")
```

### ✅ Day 14 Checklist
- [ ] Watched Strang Lecture 21 (and ideally the 3B1B video)
- [ ] I can explain: "Av = λv means A scales v by λ without changing direction"
- [ ] Verified Av = λv for computed eigenpairs
- [ ] Verified trace = Σλ and det = Πλ
- [ ] Implemented power iteration from scratch
- [ ] Know the difference between `eig` (general) and `eigh` (symmetric, faster)

---

## 📅 Day 15 — Wednesday, Aug 20: SVD (Singular Value Decomposition)

### 🌅 Morning — Watch [Strang Lecture 29](https://www.youtube.com/watch?v=TX_vooSnhm8) (~50 min)
Also: [Steve Brunton — SVD](https://www.youtube.com/watch?v=nbBvuuNVfco) (25 min) — excellent complementary explanation.

**The big idea:** A = UΣVᵀ. EVERY matrix has an SVD. It decomposes any transformation into: rotate (Vᵀ) → stretch (Σ) → rotate (U). The singular values in Σ tell you how much each direction is stretched.

### 🌙 Evening — Code

**File:** `week04_linear_algebra_part2/day15_svd.py`

```python
# Exercise 1: Compute SVD and reconstruct A
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]], dtype=float)

U, S, Vt = np.linalg.svd(A, full_matrices=False)
print(f"U shape: {U.shape}")    # (4, 3) — left singular vectors
print(f"S shape: {S.shape}")    # (3,) — singular values
print(f"Vt shape: {Vt.shape}")  # (3, 3) — right singular vectors

# Reconstruct: A = U @ diag(S) @ Vt
A_reconstructed = U @ np.diag(S) @ Vt
print(f"\nOriginal:\n{A}")
print(f"\nReconstructed:\n{A_reconstructed}")
assert np.allclose(A, A_reconstructed)
print("✅ A = UΣVᵀ verified!")

# Exercise 2: IMAGE COMPRESSION WITH SVD
print("\n" + "="*50)
print("IMAGE COMPRESSION WITH SVD")
print("="*50)

# Create a simple gradient image (or load a real one)
img = np.outer(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
# Add some structure
img += 0.3 * np.sin(np.linspace(0, 4*np.pi, 100)).reshape(100, 1)

U, S, Vt = np.linalg.svd(img, full_matrices=False)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
ranks = [1, 2, 5, 10, 20, 50]

for ax, k in zip(axes.ravel(), ranks):
    # Rank-k approximation
    img_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    
    # Compression ratio
    original_size = img.shape[0] * img.shape[1]
    compressed_size = k * (img.shape[0] + img.shape[1] + 1)
    ratio = original_size / compressed_size
    
    ax.imshow(img_approx, cmap='viridis')
    ax.set_title(f'Rank {k} (compression: {ratio:.1f}x)')
    ax.axis('off')

plt.suptitle('SVD Image Compression', fontsize=16)
plt.tight_layout()
plt.savefig('week04_linear_algebra_part2/day15_svd_compression.png', dpi=150)
plt.show()
print("✅ SVD compression plot saved!")

# Exercise 3: Verify singular values = sqrt(eigenvalues of AᵀA)
ATA = A.T @ A
eigenvalues_ATA = np.linalg.eigvalsh(ATA)  # Eigenvalues of AᵀA
eigenvalues_ATA_sorted = np.sort(eigenvalues_ATA)[::-1]  # Descending

print(f"\nSingular values:          {S}")
print(f"sqrt(eigenvalues of AᵀA): {np.sqrt(eigenvalues_ATA_sorted)}")
assert np.allclose(S, np.sqrt(eigenvalues_ATA_sorted), atol=1e-10)
print("✅ σᵢ = √(eigenvalue of AᵀA) verified!")
```

### ✅ Day 15 Checklist
- [ ] Watched Strang Lecture 29 (+ optionally Brunton's SVD video)
- [ ] I can explain: "SVD decomposes A into rotation-stretch-rotation"
- [ ] Reconstructed A from U, S, Vt
- [ ] Built image compression using truncated SVD with visualization
- [ ] Verified: singular values = sqrt(eigenvalues of AᵀA)

---

## 📅 Day 16 — Thursday, Aug 21: PCA from Scratch

### 🌅 Morning — Read/Watch about PCA
- [StatQuest: PCA](https://www.youtube.com/watch?v=FgakZw6K1QQ) (20 min — excellent visual explanation)
- Then read: your own notes on eigenvalues + SVD. PCA is just their application.

**PCA in one sentence:** "Find the directions of maximum variance in your data, and project the data onto those directions."

### 🌙 Evening — Code

**File:** `week04_linear_algebra_part2/day16_pca_from_scratch.py`

```python
import matplotlib.pyplot as plt

# Step 1: Generate data with obvious principal component
rng = np.random.default_rng(42)
n = 200

# Create elongated 2D blob (data stretches more in one direction)
mean = [3, 5]
cov = [[3, 2.5],
       [2.5, 3]]  # Positive correlation → elongated shape
data = rng.multivariate_normal(mean, cov, n)

print(f"Data shape: {data.shape}")  # (200, 2)

# Step 2: PCA via eigendecomposition
def pca_eigen(X, n_components):
    """PCA using eigendecomposition of covariance matrix"""
    # 1. Center the data (subtract mean of each feature)
    X_centered = X - X.mean(axis=0)
    
    # 2. Compute covariance matrix
    cov_matrix = (X_centered.T @ X_centered) / (X.shape[0] - 1)
    print(f"Covariance matrix:\n{cov_matrix}")
    
    # 3. Eigendecompose the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 4. Sort by descending eigenvalue (largest = most variance)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]
    
    # 5. Explained variance ratio
    explained_var_ratio = eigenvalues / eigenvalues.sum()
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Explained variance ratio: {explained_var_ratio}")
    
    # 6. Project onto top-k components
    components = eigenvectors[:, :n_components]
    X_projected = X_centered @ components
    
    return X_projected, components, explained_var_ratio

# Step 3: PCA via SVD (more numerically stable)
def pca_svd(X, n_components):
    """PCA using SVD — preferred method in practice"""
    X_centered = X - X.mean(axis=0)
    
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # The principal components are the rows of Vt
    components = Vt[:n_components].T
    X_projected = X_centered @ components
    
    # Explained variance
    explained_var = (S ** 2) / (X.shape[0] - 1)
    explained_var_ratio = explained_var / explained_var.sum()
    
    return X_projected, components, explained_var_ratio[:n_components]

# Run both
print("=== PCA via Eigendecomposition ===")
X_proj_eigen, comps_eigen, evr_eigen = pca_eigen(data, 1)

print("\n=== PCA via SVD ===")
X_proj_svd, comps_svd, evr_svd = pca_svd(data, 1)

# Verify both give the same result (up to sign flip)
assert np.allclose(np.abs(X_proj_eigen), np.abs(X_proj_svd), atol=1e-8)
print("\n✅ Both PCA methods give the same projection!")

# Step 4: VISUALIZE
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Original data
axes[0].scatter(data[:, 0], data[:, 1], alpha=0.5, s=20)
axes[0].set_title('Original 2D Data')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].axis('equal')

# Data with principal component direction shown
data_centered = data - data.mean(axis=0)
axes[1].scatter(data_centered[:, 0], data_centered[:, 1], alpha=0.5, s=20)
# Draw the principal component direction
pc_direction = comps_eigen[:, 0]
scale = 4  # For visibility
axes[1].arrow(0, 0, scale*pc_direction[0], scale*pc_direction[1],
              head_width=0.2, color='red', linewidth=3, label='PC1')
axes[1].set_title(f'Centered Data + PC1 Direction\n(explains {evr_eigen[0]*100:.1f}% of variance)')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].axis('equal')
axes[1].legend()

# Projected 1D data
axes[2].scatter(X_proj_eigen, np.zeros_like(X_proj_eigen), alpha=0.5, s=20)
axes[2].set_title('Projected onto PC1 (1D)')
axes[2].set_xlabel('PC1 score')
axes[2].set_yticks([])

plt.tight_layout()
plt.savefig('week04_linear_algebra_part2/day16_pca.png', dpi=150)
plt.show()
print("✅ PCA visualization saved!")
```

### ✅ Day 16 Checklist
- [ ] Watched StatQuest PCA video
- [ ] Implemented PCA two ways: eigendecomposition and SVD
- [ ] Both methods give the same result
- [ ] I can explain: "PCA finds directions of maximum variance"
- [ ] Computed and understand explained variance ratio
- [ ] Visualized the principal component direction on 2D data

---

## 📅 Day 17 — Friday, Aug 22: Norms, Distances & Cosine Similarity

### 🌅 Morning — Read about norms and distances
Review your notes on L1, L2, L∞ norms. These appear EVERYWHERE in ML: regularization (L1=Lasso, L2=Ridge), distance metrics (KNN, clustering), and loss functions.

### 🌙 Evening — Code

**File:** `week04_linear_algebra_part2/day17_norms_distances.py`

```python
# Exercise 1: All norms
v = np.array([3, -4, 12])
print(f"L1 norm (Manhattan):  {np.linalg.norm(v, ord=1)}")   # |3|+|-4|+|12| = 19
print(f"L2 norm (Euclidean):  {np.linalg.norm(v, ord=2)}")   # √(9+16+144) = 13
print(f"L∞ norm (max):        {np.linalg.norm(v, ord=np.inf)}")  # max(|3|,|-4|,|12|) = 12

# Exercise 2: Pairwise Euclidean distance matrix
rng = np.random.default_rng(42)
X = rng.random((100, 10))  # 100 points in 10D

# Method: broadcasting — the key ML trick
# X[:, np.newaxis, :] has shape (100, 1, 10)
# X[np.newaxis, :, :] has shape (1, 100, 10)
# Difference broadcasts to (100, 100, 10)
diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (100, 100, 10)
dist_matrix = np.sqrt(np.sum(diff**2, axis=2))      # (100, 100)

print(f"\nDistance matrix shape: {dist_matrix.shape}")
print(f"dist[0, 0] (self-distance): {dist_matrix[0, 0]:.6f}")  # Should be 0
print(f"dist[0, 1]: {dist_matrix[0, 1]:.6f}")
assert np.allclose(dist_matrix, dist_matrix.T), "Distance matrix should be symmetric!"
print("✅ Distance matrix is symmetric!")

# Exercise 3: Cosine similarity matrix (used for text embeddings)
def cosine_similarity_matrix(X):
    """Compute pairwise cosine similarity for rows of X"""
    # Normalize each row to unit length
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_normalized = X / norms
    # Cosine similarity = dot product of unit vectors
    return X_normalized @ X_normalized.T

sim_matrix = cosine_similarity_matrix(X[:20])  # Use 20 points for visualization
print(f"\nCosine similarity matrix shape: {sim_matrix.shape}")
print(f"Self-similarity (should be 1.0): {sim_matrix[0, 0]:.6f}")

# Visualize
plt.figure(figsize=(8, 6))
plt.imshow(sim_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
plt.colorbar(label='Cosine Similarity')
plt.title('Pairwise Cosine Similarity')
plt.savefig('week04_linear_algebra_part2/day17_cosine_sim.png', dpi=150)
plt.show()

# Exercise 4: K-Nearest Neighbors from scratch!
def knn_classify(X_train, y_train, X_test, k=5):
    """K-Nearest Neighbors classifier from scratch"""
    predictions = []
    for test_point in X_test:
        # Compute distance to all training points
        distances = np.sqrt(np.sum((X_train - test_point)**2, axis=1))
        # Find k nearest
        k_nearest_idx = np.argpartition(distances, k)[:k]
        k_nearest_labels = y_train[k_nearest_idx]
        # Vote (majority class)
        prediction = np.bincount(k_nearest_labels).argmax()
        predictions.append(prediction)
    return np.array(predictions)

# Test on simple 2D data
rng = np.random.default_rng(42)
# Class 0: centered around (0, 0)
X0 = rng.normal(0, 1, (50, 2))
# Class 1: centered around (3, 3)
X1 = rng.normal(3, 1, (50, 2))

X_all = np.vstack([X0, X1])
y_all = np.array([0]*50 + [1]*50)

# Split
X_tr, X_te = X_all[:80], X_all[80:]
y_tr, y_te = y_all[:80], y_all[80:]

y_pred = knn_classify(X_tr, y_tr, X_te, k=5)
accuracy = np.mean(y_pred == y_te)
print(f"\nKNN accuracy: {accuracy*100:.1f}%")
print("✅ KNN from scratch works!")
```

### ✅ Day 17 Checklist
- [ ] Computed L1, L2, L∞ norms
- [ ] Built pairwise Euclidean distance matrix using broadcasting
- [ ] Implemented cosine similarity (used for text/embedding similarity)
- [ ] Built KNN classifier from scratch using distances
- [ ] Understand: L1→Lasso, L2→Ridge, cosine→text similarity

---

## 📅 Day 18 — Saturday, Aug 23: Week 2-3 Linear Algebra Mastery Test

### Full Day — Implement ALL of these WITHOUT looking at previous code:

**File:** `week05_linear_algebra_part2_contd/day18_mastery_test.py`

1. **Gaussian elimination** with partial pivoting → solve a 5×5 system
2. **Matrix inverse** via Gauss-Jordan
3. **Power iteration** for dominant eigenvalue
4. **Gram-Schmidt** orthogonalization
5. **PCA** using SVD on a synthetic dataset with 5 features → reduce to 2
6. **K-NN** using pairwise distances on a 3-class problem
7. **SVD image compression** — create a gradient image, compress at rank 1, 5, 10, 20

**If you can do all 7 in one sitting (3-4 hours) → Linear Algebra mastery is complete.**
**If you can't → identify which ones you struggle with, redo that specific Day.**

### ✅ Day 18 Checklist
- [ ] All 7 implementations complete without reference
- [ ] All produce correct results (verified against NumPy built-ins)
- [ ] All code pushed to GitHub with clear README

---

## 🔄 Sunday, Aug 24 — REST DAY

---

# WEEK 4 — CALCULUS & OPTIMIZATION
### Aug 25 (Mon) – Aug 30 (Sat)
> *This is where neural network training comes from. Derivatives → Gradients → Gradient Descent → Backpropagation.*
> **Source:** [3Blue1Brown — Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)

---

## 📅 Day 19 — Monday, Aug 25: Derivatives & Numerical Gradients

### 🌅 Morning — Watch 3Blue1Brown Essence of Calculus
- **Video 1:** [The essence of calculus](https://www.youtube.com/watch?v=WUvTyaaNkzM) (17 min)
- **Video 2:** [The paradox of the derivative](https://www.youtube.com/watch?v=9vKqVkMQHKk) (17 min)
- **Video 3:** [Derivative formulas through geometry](https://www.youtube.com/watch?v=S0_qX4VJhMQ) (18 min)
- **Video 4:** [Chain rule](https://www.youtube.com/watch?v=YG15m2VwSjA) (10 min) — **MOST IMPORTANT FOR DEEP LEARNING**

### 🌙 Evening — Code

**File:** `week06_calculus/day19_derivatives.py`

```python
# THE NUMERICAL GRADIENT — your debugging tool for ALL of deep learning
def numerical_gradient(f, x, eps=1e-5):
    """
    Compute the gradient of f at x using central differences.
    Works for scalar functions of vector inputs.
    
    This is HOW you verify your analytical gradients are correct.
    Every ML researcher uses this as a sanity check.
    """
    grad = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        x_plus = x.copy().astype(float)
        x_minus = x.copy().astype(float)
        x_plus.flat[i] += eps
        x_minus.flat[i] -= eps
        grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

# Test 1: f(x) = x³ - 2x + 1 at x = 3
# Analytical derivative: f'(x) = 3x² - 2, f'(3) = 25
f = lambda x: x[0]**3 - 2*x[0] + 1
x = np.array([3.0])
grad = numerical_gradient(f, x)
print(f"Numerical gradient of x³-2x+1 at x=3: {grad[0]:.6f}")
print(f"Analytical answer: 25")
assert abs(grad[0] - 25) < 1e-4, "Gradient is wrong!"
print("✅ Numerical gradient matches analytical!")

# Test 2: f(x, y) = x²y + y³ at (1, 2)
# ∂f/∂x = 2xy = 4, ∂f/∂y = x² + 3y² = 13
f2 = lambda xy: xy[0]**2 * xy[1] + xy[1]**3
x2 = np.array([1.0, 2.0])
grad2 = numerical_gradient(f2, x2)
print(f"\nGradient of x²y + y³ at (1,2): {grad2}")
print(f"Analytical: [4, 13]")
assert np.allclose(grad2, [4, 13], atol=1e-4)
print("✅ Multivariable gradient correct!")

# Test 3: Verify chain rule numerically
# d/dx sin(x²) at x=1
# Chain rule: cos(x²) × 2x = cos(1) × 2 ≈ 1.0806
f3 = lambda x: np.sin(x[0]**2)
x3 = np.array([1.0])
grad3 = numerical_gradient(f3, x3)
analytical3 = np.cos(1.0) * 2.0
print(f"\nd/dx sin(x²) at x=1:")
print(f"  Numerical:  {grad3[0]:.6f}")
print(f"  Analytical: {analytical3:.6f}")
assert abs(grad3[0] - analytical3) < 1e-4
print("✅ Chain rule verified numerically!")
```

### ✅ Day 19 Checklist
- [ ] Watched 3B1B calculus videos 1-4
- [ ] Built `numerical_gradient` function — **YOU WILL USE THIS FOR THE REST OF YOUR CAREER**
- [ ] Verified it on single-variable, multi-variable, and chain rule cases
- [ ] Understand: "derivative = rate of change = slope of tangent"
- [ ] Understand chain rule: d/dx f(g(x)) = f'(g(x)) × g'(x)

---

## 📅 Day 20 — Tuesday, Aug 26: Gradient Descent

### 🌅 Morning — Watch:
- **3B1B Video 7:** [Limits](https://www.youtube.com/watch?v=kfF40MiS7zA) (18 min)
- **3B1B Video 10:** [Taylor series](https://www.youtube.com/watch?v=3d6DsjIBzJ4) (22 min)
- Read about gradient descent intuitively: gradient points uphill → negative gradient points downhill → follow negative gradient to minimize

### 🌙 Evening — Code

**File:** `week06_calculus/day20_gradient_descent.py`

```python
import matplotlib.pyplot as plt

# Exercise 1: Minimize f(x) = (x - 3)² + 1
def f_simple(x):
    return (x - 3)**2 + 1

def grad_simple(x):
    return 2 * (x - 3)

def gradient_descent_1d(f, grad, x0, lr, n_iters):
    """1D gradient descent with history tracking"""
    x = x0
    history = [x]
    loss_history = [f(x)]
    
    for i in range(n_iters):
        g = grad(x)
        x = x - lr * g
        history.append(x)
        loss_history.append(f(x))
    
    return x, history, loss_history

x_final, hist, loss_hist = gradient_descent_1d(f_simple, grad_simple, x0=10.0, lr=0.1, n_iters=50)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: x vs iteration
ax1.plot(hist, linewidth=2)
ax1.axhline(y=3, color='r', linestyle='--', label='Optimum (x=3)')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('x')
ax1.set_title('x converging to optimum')
ax1.legend()

# Plot 2: loss vs iteration
ax2.plot(loss_hist, linewidth=2)
ax2.axhline(y=1, color='r', linestyle='--', label='Min loss (1.0)')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss')
ax2.set_title('Loss decreasing')
ax2.legend()

plt.tight_layout()
plt.savefig('week06_calculus/day20_gradient_descent_1d.png', dpi=150)
plt.show()

print(f"Final x: {x_final:.6f} (should be ~3.0)")
print(f"Final loss: {f_simple(x_final):.6f} (should be ~1.0)")

# Exercise 2: Learning rate experiment
print("\n=== Learning Rate Experiment ===")
learning_rates = [0.001, 0.01, 0.1, 0.5, 0.9, 1.1]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, lr in zip(axes.ravel(), learning_rates):
    try:
        _, hist, _ = gradient_descent_1d(f_simple, grad_simple, x0=10.0, lr=lr, n_iters=50)
        ax.plot(hist, linewidth=2)
        ax.axhline(y=3, color='r', linestyle='--', alpha=0.5)
        ax.set_title(f'lr = {lr}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('x')
        
        if abs(hist[-1] - 3) > 100:
            ax.set_title(f'lr = {lr} (DIVERGED!)', color='red')
    except:
        ax.set_title(f'lr = {lr} (DIVERGED!)', color='red')

plt.suptitle('Effect of Learning Rate', fontsize=14)
plt.tight_layout()
plt.savefig('week06_calculus/day20_learning_rate_experiment.png', dpi=150)
plt.show()

# Exercise 3: 2D gradient descent with contour plot
def f_2d(xy):
    x, y = xy
    return x**2 + y**2

def grad_2d(xy):
    x, y = xy
    return np.array([2*x, 2*y])

def gradient_descent_2d(grad, xy0, lr, n_iters):
    xy = np.array(xy0, dtype=float)
    history = [xy.copy()]
    for _ in range(n_iters):
        xy = xy - lr * grad(xy)
        history.append(xy.copy())
    return xy, np.array(history)

xy_final, path = gradient_descent_2d(grad_2d, [5.0, 5.0], lr=0.1, n_iters=50)

# Contour plot
x_grid = np.linspace(-6, 6, 100)
y_grid = np.linspace(-6, 6, 100)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
Z_grid = X_grid**2 + Y_grid**2

plt.figure(figsize=(8, 8))
plt.contour(X_grid, Y_grid, Z_grid, levels=20, cmap='viridis', alpha=0.6)
plt.colorbar(label='f(x,y) = x² + y²')
plt.plot(path[:, 0], path[:, 1], 'ro-', markersize=3, linewidth=1, label='GD path')
plt.plot(path[0, 0], path[0, 1], 'gs', markersize=10, label='Start')
plt.plot(path[-1, 0], path[-1, 1], 'r*', markersize=15, label='End')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Gradient Descent')
plt.legend()
plt.savefig('week06_calculus/day20_gd_2d.png', dpi=150)
plt.show()

print(f"Final position: ({xy_final[0]:.6f}, {xy_final[1]:.6f})")
print(f"Should be near (0, 0)")
```

### ✅ Day 20 Checklist
- [ ] Watched 3B1B calculus videos
- [ ] Implemented 1D gradient descent with convergence visualization
- [ ] Experimented with learning rates (too small → slow, too big → diverge)
- [ ] Implemented 2D gradient descent with contour plot
- [ ] Understand: "gradient descent walks downhill by following the negative gradient"

---

## 📅 Day 21 — Wednesday, Aug 27: Activation Functions & Their Gradients

### 🌅 Morning (6:00 – 7:30 AM) — Read & Watch

**What to watch:** 3B1B Neural Networks Ch.3 — focus on how activation functions shape the network
**What to read:** https://en.wikipedia.org/wiki/Activation_function — study the shapes and derivatives

**Key concept:** Every activation function has a DERIVATIVE. You will need these derivatives for backpropagation. Write each on paper.

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week04_calculus/day21_activation_functions.py`

**Exercise 1:** Implement ALL major activation functions and their derivatives.
```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)

def softmax(x):
    """Numerically stable softmax"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Test all functions
x = np.linspace(-5, 5, 1000)
print(f"sigmoid(0) = {sigmoid(0):.4f}")     # Should be 0.5
print(f"tanh(0) = {tanh(0):.4f}")           # Should be 0.0
print(f"relu(-2) = {relu(-2):.4f}")         # Should be 0.0
print(f"relu(3) = {relu(3):.4f}")           # Should be 3.0
```

**Exercise 2:** Verify derivatives numerically.
```python
def numerical_gradient(f, x, epsilon=1e-7):
    """Compute df/dx using central difference"""
    return (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)

# Test sigmoid derivative
x_test = np.array([-2.0, 0.0, 1.0, 3.0])
analytical = sigmoid_derivative(x_test)
numerical = numerical_gradient(sigmoid, x_test)
print(f"Sigmoid analytical: {analytical}")
print(f"Sigmoid numerical:  {numerical}")
assert np.allclose(analytical, numerical, atol=1e-5), "Sigmoid derivative is wrong!"
print("✅ Sigmoid derivative verified!")

# Test tanh derivative
analytical_t = tanh_derivative(x_test)
numerical_t = numerical_gradient(tanh, x_test)
assert np.allclose(analytical_t, numerical_t, atol=1e-5), "Tanh derivative is wrong!"
print("✅ Tanh derivative verified!")

# Test ReLU derivative
analytical_r = relu_derivative(x_test)
numerical_r = numerical_gradient(relu, x_test)
# Note: ReLU is not differentiable at 0, but works elsewhere
assert np.allclose(analytical_r[x_test != 0], numerical_r[x_test != 0], atol=1e-5)
print("✅ ReLU derivative verified!")
```

**Exercise 3:** Plot all activations and their derivatives (2×4 subplot grid).
```python
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
x = np.linspace(-5, 5, 1000)

funcs = [
    ("Sigmoid", sigmoid, sigmoid_derivative),
    ("Tanh", tanh, tanh_derivative),
    ("ReLU", relu, relu_derivative),
    ("Leaky ReLU", leaky_relu, leaky_relu_derivative),
]

for i, (name, f, df) in enumerate(funcs):
    axes[0, i].plot(x, f(x), 'b-', linewidth=2)
    axes[0, i].set_title(f"{name}", fontsize=12)
    axes[0, i].axhline(y=0, color='gray', linewidth=0.5)
    axes[0, i].axvline(x=0, color='gray', linewidth=0.5)
    axes[0, i].grid(True, alpha=0.3)

    axes[1, i].plot(x, df(x), 'r-', linewidth=2)
    axes[1, i].set_title(f"{name} Derivative", fontsize=12)
    axes[1, i].axhline(y=0, color='gray', linewidth=0.5)
    axes[1, i].axvline(x=0, color='gray', linewidth=0.5)
    axes[1, i].grid(True, alpha=0.3)

axes[0, 0].set_ylabel("f(x)")
axes[1, 0].set_ylabel("f'(x)")
plt.tight_layout()
plt.savefig('week04_calculus/day21_activations.png', dpi=150)
plt.show()
print("✅ Activation plot saved!")
```

**Exercise 4:** Demonstrate the vanishing gradient problem with sigmoid.
```python
# Stack 10 sigmoid layers — what happens to the gradient?
x = np.array([1.0])
gradient = 1.0

print("Layer | Output | Local Gradient | Cumulative Gradient")
print("-" * 55)
for layer in range(10):
    x = sigmoid(x)
    local_grad = sigmoid_derivative(x)
    gradient *= local_grad[0]
    print(f"  {layer+1:2d}  |  {x[0]:.6f}  |    {local_grad[0]:.6f}     |    {gradient:.10f}")

print(f"\nAfter 10 sigmoid layers, gradient = {gradient:.2e}")
print("This is the VANISHING GRADIENT problem!")
print("ReLU fixes this because its derivative is 1 for positive values.")
```

### ✅ Day 21 Checklist
- [ ] I can write sigmoid, tanh, ReLU, leaky ReLU, and softmax from memory
- [ ] I can write the DERIVATIVE of each from memory
- [ ] I verified derivatives numerically using central difference
- [ ] I understand the vanishing gradient problem and why ReLU helps
- [ ] Code is pushed to GitHub

---

## 📅 Day 22 — Thursday, Aug 28: Loss Functions & Their Gradients

### 🌅 Morning (6:00 – 7:30 AM) — Read

**What to read:** "Loss Functions in Machine Learning" — understand WHY each loss function exists
- MSE for regression, Cross-Entropy for classification
- Why cross-entropy + softmax work together mathematically

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week04_calculus/day22_loss_functions.py`

**Exercise 1:** Implement MSE loss and its gradient.
```python
import numpy as np

def mse_loss(y_pred, y_true):
    """Mean Squared Error: L = (1/n) * Σ(y_pred - y_true)²"""
    return np.mean((y_pred - y_true) ** 2)

def mse_gradient(y_pred, y_true):
    """dL/dy_pred = (2/n) * (y_pred - y_true)"""
    n = len(y_true)
    return (2 / n) * (y_pred - y_true)

# Test
y_true = np.array([1.0, 0.0, 1.0, 0.0])
y_pred = np.array([0.9, 0.1, 0.8, 0.3])

loss = mse_loss(y_pred, y_true)
grad = mse_gradient(y_pred, y_true)
print(f"MSE Loss: {loss:.6f}")
print(f"MSE Gradient: {grad}")

# Verify gradient numerically
epsilon = 1e-7
numerical_grad = np.zeros_like(y_pred)
for i in range(len(y_pred)):
    y_plus = y_pred.copy(); y_plus[i] += epsilon
    y_minus = y_pred.copy(); y_minus[i] -= epsilon
    numerical_grad[i] = (mse_loss(y_plus, y_true) - mse_loss(y_minus, y_true)) / (2 * epsilon)

assert np.allclose(grad, numerical_grad, atol=1e-5), "MSE gradient is wrong!"
print("✅ MSE gradient verified numerically!")
```

**Exercise 2:** Implement binary cross-entropy loss and its gradient.
```python
def binary_cross_entropy(y_pred, y_true, epsilon=1e-15):
    """BCE = -(1/n) * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Prevent log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def bce_gradient(y_pred, y_true, epsilon=1e-15):
    """dL/dŷ = -(1/n) * [y/ŷ - (1-y)/(1-ŷ)]"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    n = len(y_true)
    return -(1 / n) * (y_true / y_pred - (1 - y_true) / (1 - y_pred))

# Test
y_true = np.array([1.0, 0.0, 1.0, 0.0])
y_pred = np.array([0.9, 0.1, 0.8, 0.3])
loss = binary_cross_entropy(y_pred, y_true)
grad = bce_gradient(y_pred, y_true)
print(f"\nBCE Loss: {loss:.6f}")
print(f"BCE Gradient: {grad}")
```

**Exercise 3:** Implement categorical cross-entropy with softmax.
```python
def softmax(logits):
    exp_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def cross_entropy_loss(logits, y_true_onehot):
    """CE = -(1/n) * Σ y_true * log(softmax(logits))"""
    probs = softmax(logits)
    probs = np.clip(probs, 1e-15, 1.0)
    return -np.mean(np.sum(y_true_onehot * np.log(probs), axis=-1))

def cross_entropy_gradient(logits, y_true_onehot):
    """The beautiful result: dL/d(logits) = softmax(logits) - y_true"""
    probs = softmax(logits)
    n = y_true_onehot.shape[0]
    return (probs - y_true_onehot) / n

# Test with 3 samples, 4 classes
logits = np.array([[2.0, 1.0, 0.1, 0.5],
                    [0.1, 3.0, 0.2, 0.1],
                    [0.5, 0.2, 2.5, 0.3]])

y_true = np.array([[1, 0, 0, 0],   # Class 0
                    [0, 1, 0, 0],   # Class 1
                    [0, 0, 1, 0]])  # Class 2

loss = cross_entropy_loss(logits, y_true)
grad = cross_entropy_gradient(logits, y_true)
print(f"\nCross-Entropy Loss: {loss:.6f}")
print(f"Gradient:\n{grad}")
print(f"Gradient shape: {grad.shape}")  # Should be (3, 4)

# THE KEY INSIGHT: gradient = softmax(logits) - y_true
# This is why softmax + cross-entropy is the standard combo!
probs = softmax(logits)
simple_grad = (probs - y_true) / 3
assert np.allclose(grad, simple_grad), "Gradient formula doesn't match!"
print("✅ Cross-entropy gradient = softmax - y_true (verified!)")
```

### ✅ Day 22 Checklist
- [ ] I can implement MSE, BCE, and categorical cross-entropy from scratch
- [ ] I can compute the gradient of each loss function
- [ ] I verified all gradients numerically
- [ ] I understand: softmax + cross-entropy gradient = softmax(logits) - y_true
- [ ] Code is pushed to GitHub

---

## 📅 Day 23 — Friday, Aug 29: Optimizers — SGD with Momentum

### 🌅 Morning (6:00 – 7:30 AM) — Read

**What to watch:** "But what is a neural network?" — 3B1B Ch. 4 (Gradient Descent in context)
**What to read:** Sebastian Ruder's blog "An overview of gradient descent optimization algorithms"

**Key concept — Momentum:**
> Plain SGD oscillates in narrow valleys. Momentum adds "velocity" — it keeps moving in the direction it was already going, like a ball rolling downhill.

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week04_calculus/day23_optimizers_momentum.py`

**Exercise 1:** Compare vanilla SGD vs SGD with Momentum on a hard function.
```python
import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(xy):
    """Rosenbrock function — a classic hard optimization surface"""
    x, y = xy
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(xy):
    x, y = xy
    dx = -2*(1 - x) + 200*(y - x**2)*(-2*x)
    dy = 200*(y - x**2)
    return np.array([dx, dy])

# Vanilla SGD
def sgd(grad_fn, xy0, lr, n_iters):
    xy = np.array(xy0, dtype=float)
    history = [xy.copy()]
    for _ in range(n_iters):
        g = grad_fn(xy)
        xy = xy - lr * g
        history.append(xy.copy())
    return xy, np.array(history)

# SGD with Momentum
def sgd_momentum(grad_fn, xy0, lr, momentum, n_iters):
    xy = np.array(xy0, dtype=float)
    velocity = np.zeros_like(xy)
    history = [xy.copy()]
    for _ in range(n_iters):
        g = grad_fn(xy)
        velocity = momentum * velocity - lr * g  # Update velocity
        xy = xy + velocity                        # Update position
        history.append(xy.copy())
    return xy, np.array(history)

# Run both from same starting point
start = [-1.0, 1.0]
_, path_sgd = sgd(rosenbrock_grad, start, lr=0.001, n_iters=5000)
_, path_mom = sgd_momentum(rosenbrock_grad, start, lr=0.001, momentum=0.9, n_iters=5000)

print(f"Vanilla SGD final: ({path_sgd[-1][0]:.4f}, {path_sgd[-1][1]:.4f})")
print(f"Momentum final:    ({path_mom[-1][0]:.4f}, {path_mom[-1][1]:.4f})")
print(f"Target: (1.0, 1.0)")

# Plot contour with both paths
x_grid = np.linspace(-2, 2, 200)
y_grid = np.linspace(-1, 3, 200)
X, Y = np.meshgrid(x_grid, y_grid)
Z = (1 - X)**2 + 100 * (Y - X**2)**2

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
plt.plot(path_sgd[:500, 0], path_sgd[:500, 1], 'r.-', markersize=1, linewidth=0.5, label='Vanilla SGD')
plt.plot(1, 1, 'g*', markersize=15)
plt.title('Vanilla SGD (first 500 steps)')
plt.legend()

plt.subplot(1, 2, 2)
plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
plt.plot(path_mom[:500, 0], path_mom[:500, 1], 'b.-', markersize=1, linewidth=0.5, label='SGD + Momentum')
plt.plot(1, 1, 'g*', markersize=15)
plt.title('SGD + Momentum (first 500 steps)')
plt.legend()

plt.tight_layout()
plt.savefig('week04_calculus/day23_momentum.png', dpi=150)
plt.show()
```

**Exercise 2:** Implement Nesterov Momentum (lookahead variant).
```python
def sgd_nesterov(grad_fn, xy0, lr, momentum, n_iters):
    xy = np.array(xy0, dtype=float)
    velocity = np.zeros_like(xy)
    history = [xy.copy()]
    for _ in range(n_iters):
        lookahead = xy + momentum * velocity  # Look ahead
        g = grad_fn(lookahead)                 # Compute gradient at lookahead
        velocity = momentum * velocity - lr * g
        xy = xy + velocity
        history.append(xy.copy())
    return xy, np.array(history)

_, path_nest = sgd_nesterov(rosenbrock_grad, start, lr=0.001, momentum=0.9, n_iters=5000)
print(f"Nesterov final: ({path_nest[-1][0]:.4f}, {path_nest[-1][1]:.4f})")
```

### ✅ Day 23 Checklist
- [ ] I implemented vanilla SGD and can explain it in one sentence
- [ ] I implemented SGD with Momentum and understand the "velocity" concept
- [ ] I implemented Nesterov Momentum and know the "lookahead" difference
- [ ] I visualized all three on the Rosenbrock function
- [ ] Code is pushed to GitHub

---

## 📅 Day 24 — Saturday, Aug 30: Adam Optimizer from Scratch

### 🌅 Morning (6:00 – 7:30 AM) — Read

**What to read:** Original Adam paper (Kingma & Ba 2014) — at least the Algorithm 1 pseudocode
**Key idea:** Adam = Momentum (first moment) + RMSProp (second moment) + bias correction

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week04_calculus/day24_adam_optimizer.py`

**Exercise 1:** Implement Adam from scratch.
```python
import numpy as np
import matplotlib.pyplot as plt

def adam_optimizer(grad_fn, xy0, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, n_iters=5000):
    xy = np.array(xy0, dtype=float)
    m = np.zeros_like(xy)  # First moment (mean of gradients)
    v = np.zeros_like(xy)  # Second moment (mean of squared gradients)
    history = [xy.copy()]

    for t in range(1, n_iters + 1):
        g = grad_fn(xy)

        # Update biased moments
        m = beta1 * m + (1 - beta1) * g         # Momentum-like
        v = beta2 * v + (1 - beta2) * g**2       # RMSProp-like

        # Bias correction (critical for early steps!)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # Update parameters
        xy = xy - lr * m_hat / (np.sqrt(v_hat) + epsilon)
        history.append(xy.copy())

    return xy, np.array(history)

# Test on Rosenbrock
def rosenbrock_grad(xy):
    x, y = xy
    dx = -2*(1 - x) + 200*(y - x**2)*(-2*x)
    dy = 200*(y - x**2)
    return np.array([dx, dy])

start = [-1.0, 1.0]
xy_final, path_adam = adam_optimizer(rosenbrock_grad, start, lr=0.01, n_iters=10000)
print(f"Adam final: ({xy_final[0]:.6f}, {xy_final[1]:.6f})")
print(f"Target:     (1.0, 1.0)")
print(f"Distance from target: {np.linalg.norm(xy_final - np.array([1.0, 1.0])):.6f}")
```

**Exercise 2:** Compare ALL optimizers side by side.
```python
from day23_optimizers_momentum import sgd, sgd_momentum, sgd_nesterov

_, path_sgd = sgd(rosenbrock_grad, start, lr=0.001, n_iters=10000)
_, path_mom = sgd_momentum(rosenbrock_grad, start, lr=0.001, momentum=0.9, n_iters=10000)
_, path_nest = sgd_nesterov(rosenbrock_grad, start, lr=0.001, momentum=0.9, n_iters=10000)
_, path_adam = adam_optimizer(rosenbrock_grad, start, lr=0.01, n_iters=10000)

# Plot loss curves
def rosenbrock(xy):
    x, y = xy
    return (1 - x)**2 + 100 * (y - x**2)**2

losses_sgd = [rosenbrock(p) for p in path_sgd]
losses_mom = [rosenbrock(p) for p in path_mom]
losses_nest = [rosenbrock(p) for p in path_nest]
losses_adam = [rosenbrock(p) for p in path_adam]

plt.figure(figsize=(10, 6))
plt.semilogy(losses_sgd[:2000], label='Vanilla SGD', alpha=0.8)
plt.semilogy(losses_mom[:2000], label='Momentum', alpha=0.8)
plt.semilogy(losses_nest[:2000], label='Nesterov', alpha=0.8)
plt.semilogy(losses_adam[:2000], label='Adam', alpha=0.8, linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Loss (log scale)')
plt.title('Optimizer Comparison on Rosenbrock Function')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('week04_calculus/day24_optimizer_comparison.png', dpi=150)
plt.show()
print("✅ Adam converges MUCH faster!")
```

### ✅ Day 24 Checklist
- [ ] I implemented Adam optimizer from scratch (m, v, bias correction)
- [ ] I can explain: m = momentum (direction), v = RMSProp (step size adaptation)
- [ ] I compared all 4 optimizers on the same function
- [ ] I understand why Adam is the default choice for most deep learning
- [ ] Code is pushed to GitHub

---

> 🌙 **Sunday Aug 31 — REST DAY.** Do not code. Review your notes from Week 4.

---

# 📦 MONTH 2: BACKPROPAGATION + PROBABILITY
## September 1 – September 30, 2026

---

# WEEK 5 — BACKPROPAGATION & NEURAL NETWORKS FROM SCRATCH
### Sep 1 (Mon) – Sep 6 (Sat)
> *This is THE week. You build a neural network from scratch — forward pass, backward pass, training loop. Everything you've learned converges here.*

---

## 📅 Day 25 — Monday, Sep 1: Computational Graphs & the Chain Rule

### 🌅 Morning (6:00 – 7:30 AM) — Watch

**What to watch:** 3B1B "Backpropagation" + "Backpropagation calculus" (2 videos)
**Also read:** Andrej Karpathy's "Yes you should understand backprop" blog post

**The chain rule in one line:**
> If `L = f(g(h(x)))`, then `dL/dx = dL/df · df/dg · dg/dh · dh/dx`
> Each layer passes its local gradient backward. That's backpropagation.

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week05_backprop/day25_computational_graph.py`

**Exercise 1:** Build a simple computational graph by hand.
```python
import numpy as np

# Compute f = (x + y) * z and its gradients
# Computational graph:
#   q = x + y
#   f = q * z

x, y, z = 2.0, 3.0, -4.0

# Forward pass
q = x + y    # q = 5
f = q * z    # f = -20

# Backward pass (chain rule)
df_df = 1.0          # dL/df = 1 (we're differentiating f w.r.t. itself)
df_dq = z            # ∂f/∂q = z = -4
df_dz = q            # ∂f/∂z = q = 5
df_dx = df_dq * 1.0  # dq/dx = 1, so df/dx = df/dq * dq/dx = -4
df_dy = df_dq * 1.0  # dq/dy = 1, so df/dy = df/dq * dq/dy = -4

print(f"f = {f}")
print(f"df/dx = {df_dx}")  # -4
print(f"df/dy = {df_dy}")  # -4
print(f"df/dz = {df_dz}")  # 5

# Verify numerically
eps = 1e-7
num_dx = ((x+eps + y) * z - (x-eps + y) * z) / (2*eps)
num_dy = ((x + y+eps) * z - (x + y-eps) * z) / (2*eps)
num_dz = ((x + y) * (z+eps) - (x + y) * (z-eps)) / (2*eps)
print(f"\nNumerical: df/dx={num_dx:.4f}, df/dy={num_dy:.4f}, df/dz={num_dz:.4f}")
print("✅ All match!")
```

**Exercise 2:** Backprop through a sigmoid neuron.
```python
# A single neuron: output = sigmoid(w·x + b)
w = np.array([0.5, -0.3, 0.8])
x = np.array([1.0, 2.0, 3.0])
b = 0.1

# Forward pass
z = np.dot(w, x) + b           # z = 0.5 - 0.6 + 2.4 + 0.1 = 2.4
a = 1 / (1 + np.exp(-z))       # sigmoid(2.4)

# Backward pass
da_dz = a * (1 - a)            # Sigmoid derivative
dz_dw = x                      # ∂z/∂w = x
dz_db = 1.0                    # ∂z/∂b = 1
dz_dx = w                      # ∂z/∂x = w

# Chain rule: dL/dw = dL/da · da/dz · dz/dw
# Assume dL/da = 1 for now (just computing partial derivatives)
dL_dw = da_dz * dz_dw
dL_db = da_dz * dz_db
dL_dx = da_dz * dz_dx

print(f"z = {z:.4f}")
print(f"a = sigmoid(z) = {a:.4f}")
print(f"dL/dw = {dL_dw}")
print(f"dL/db = {dL_db:.4f}")
print(f"dL/dx = {dL_dx}")
```

**Exercise 3:** Backprop through a 2-layer computation.
```python
# f = sigmoid(w2 * relu(w1 * x + b1) + b2)
x_val = 1.5
w1, b1 = 0.8, -0.2
w2, b2 = 1.2, 0.5

# Forward pass
z1 = w1 * x_val + b1           # Linear 1
a1 = max(0, z1)                # ReLU
z2 = w2 * a1 + b2              # Linear 2
a2 = 1 / (1 + np.exp(-z2))     # Sigmoid

print(f"Forward: x={x_val} → z1={z1:.4f} → a1={a1:.4f} → z2={z2:.4f} → a2={a2:.4f}")

# Backward pass
da2_dz2 = a2 * (1 - a2)
dz2_da1 = w2
dz2_dw2 = a1
dz2_db2 = 1.0
da1_dz1 = 1.0 if z1 > 0 else 0.0  # ReLU derivative
dz1_dw1 = x_val
dz1_db1 = 1.0

# Chain rule
dL_dw2 = da2_dz2 * dz2_dw2
dL_db2 = da2_dz2 * dz2_db2
dL_dw1 = da2_dz2 * dz2_da1 * da1_dz1 * dz1_dw1
dL_db1 = da2_dz2 * dz2_da1 * da1_dz1 * dz1_db1

print(f"\nGradients:")
print(f"  dL/dw2 = {dL_dw2:.6f}")
print(f"  dL/db2 = {dL_db2:.6f}")
print(f"  dL/dw1 = {dL_dw1:.6f}")
print(f"  dL/db1 = {dL_db1:.6f}")
```

### ✅ Day 25 Checklist
- [ ] I can draw a computational graph for any expression
- [ ] I can compute gradients backward through the graph using the chain rule
- [ ] I verified all gradients numerically
- [ ] I understand: backprop = repeated application of the chain rule
- [ ] Code is pushed to GitHub

---

## 📅 Day 26 — Tuesday, Sep 2: Forward Pass — 2-Layer Neural Network

### 🌅 Morning (6:00 – 7:30 AM) — Read

**What to read:** Review your notes on matrix multiplication and activation functions.
**Key architecture:** Input(n_features) → Dense(hidden_size, ReLU) → Dense(n_classes, Softmax)

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week05_backprop/day26_forward_pass.py`

**Exercise 1:** Implement a 2-layer network forward pass.
```python
import numpy as np

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, X):
        """
        X: (batch_size, input_size)
        Returns: probabilities (batch_size, output_size)
        """
        # Layer 1: Linear + ReLU
        self.z1 = X @ self.W1 + self.b1          # (batch, hidden)
        self.a1 = self.relu(self.z1)               # (batch, hidden)

        # Layer 2: Linear + Softmax
        self.z2 = self.a1 @ self.W2 + self.b2     # (batch, output)
        self.a2 = self.softmax(self.z2)            # (batch, output)

        # Cache input for backward pass
        self.X = X
        return self.a2

# Test
net = TwoLayerNet(input_size=784, hidden_size=128, output_size=10)

# Fake batch of 32 "images" (flattened 28×28)
X_batch = np.random.randn(32, 784)
probs = net.forward(X_batch)

print(f"Input shape:  {X_batch.shape}")    # (32, 784)
print(f"Output shape: {probs.shape}")      # (32, 10)
print(f"Sum of probs for sample 0: {probs[0].sum():.6f}")  # Should be 1.0
print(f"Predicted class for sample 0: {np.argmax(probs[0])}")

# Verify all rows sum to 1
assert np.allclose(probs.sum(axis=1), 1.0), "Softmax rows don't sum to 1!"
print("✅ All probability rows sum to 1!")
```

**Exercise 2:** Compute cross-entropy loss.
```python
def compute_loss(probs, y_true):
    """
    probs: (batch_size, n_classes) — softmax output
    y_true: (batch_size,) — integer class labels
    """
    batch_size = len(y_true)
    # Select probability of correct class for each sample
    correct_probs = probs[np.arange(batch_size), y_true]
    # Cross-entropy loss
    loss = -np.mean(np.log(np.clip(correct_probs, 1e-15, 1.0)))
    return loss

# Fake labels
y_batch = np.random.randint(0, 10, size=32)
loss = compute_loss(probs, y_batch)
print(f"\nCross-entropy loss: {loss:.4f}")
print(f"Random guess loss should be ~{-np.log(1/10):.4f} (= -log(1/10))")
# With random weights, loss should be around 2.3 (= -log(0.1))
```

### ✅ Day 26 Checklist
- [ ] I implemented a 2-layer network forward pass with matrix operations
- [ ] I understand the shapes at every step: (batch, input) → (batch, hidden) → (batch, output)
- [ ] I verified softmax outputs sum to 1
- [ ] I computed cross-entropy loss and it's ~2.3 for random weights (correct!)
- [ ] Code is pushed to GitHub

---

## 📅 Day 27 — Wednesday, Sep 3: Backward Pass — Backpropagation

### 🌅 Morning (6:00 – 7:30 AM) — Derive

**DO THIS ON PAPER.** Derive the gradient for each parameter:
```
Forward: z1 = XW1+b1 → a1 = relu(z1) → z2 = a1·W2+b2 → a2 = softmax(z2) → L = CE(a2, y)

Backward:
  dL/dz2 = a2 - y_onehot           (softmax + CE shortcut)
  dL/dW2 = a1.T @ dL/dz2
  dL/db2 = sum(dL/dz2, axis=0)
  dL/da1 = dL/dz2 @ W2.T
  dL/dz1 = dL/da1 * relu'(z1)
  dL/dW1 = X.T @ dL/dz1
  dL/db1 = sum(dL/dz1, axis=0)
```

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week05_backprop/day27_backward_pass.py`

**Exercise 1:** Add backward method to TwoLayerNet.
```python
import numpy as np

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, y_true):
        """
        Compute gradients for all parameters.
        y_true: (batch_size,) — integer class labels
        """
        batch_size = len(y_true)

        # Convert labels to one-hot
        y_onehot = np.zeros_like(self.a2)
        y_onehot[np.arange(batch_size), y_true] = 1.0

        # Output layer gradient (softmax + CE shortcut)
        dz2 = (self.a2 - y_onehot) / batch_size   # (batch, output)

        # Layer 2 gradients
        self.dW2 = self.a1.T @ dz2                 # (hidden, output)
        self.db2 = np.sum(dz2, axis=0)             # (output,)

        # Backprop to hidden layer
        da1 = dz2 @ self.W2.T                      # (batch, hidden)
        dz1 = da1 * (self.z1 > 0).astype(float)   # ReLU derivative

        # Layer 1 gradients
        self.dW1 = self.X.T @ dz1                  # (input, hidden)
        self.db1 = np.sum(dz1, axis=0)             # (hidden,)

# Test: verify with numerical gradient checking
net = TwoLayerNet(4, 8, 3)
X = np.random.randn(5, 4)
y = np.array([0, 1, 2, 1, 0])

# Analytical gradients
probs = net.forward(X)
net.backward(y)

# Numerical gradient for W1
def loss_fn(W1_flat):
    net_copy_W1 = W1_flat.reshape(net.W1.shape)
    old_W1 = net.W1.copy()
    net.W1 = net_copy_W1
    p = net.forward(X)
    L = -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0)))
    net.W1 = old_W1
    return L

eps = 1e-5
numerical_dW1 = np.zeros_like(net.W1)
for i in range(net.W1.shape[0]):
    for j in range(net.W1.shape[1]):
        old = net.W1[i, j]
        net.W1[i, j] = old + eps
        loss_plus = -np.mean(np.log(np.clip(net.forward(X)[np.arange(len(y)), y], 1e-15, 1.0)))
        net.W1[i, j] = old - eps
        loss_minus = -np.mean(np.log(np.clip(net.forward(X)[np.arange(len(y)), y], 1e-15, 1.0)))
        net.W1[i, j] = old
        numerical_dW1[i, j] = (loss_plus - loss_minus) / (2 * eps)

net.forward(X)
net.backward(y)

diff = np.max(np.abs(net.dW1 - numerical_dW1))
print(f"Max difference between analytical and numerical dW1: {diff:.2e}")
assert diff < 1e-4, f"Gradient check FAILED! diff={diff}"
print("✅ Gradient check PASSED for W1!")
```

### ✅ Day 27 Checklist
- [ ] I derived all gradient formulas on paper
- [ ] I implemented backward() with correct matrix shapes
- [ ] I verified gradients numerically (gradient checking)
- [ ] I understand: dL/dz2 = softmax - y_onehot (the key formula)
- [ ] Code is pushed to GitHub

---

## 📅 Day 28 — Thursday, Sep 4: Full Training Loop

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week05_backprop/day28_training_loop.py`

**Exercise 1:** Add update method and train on synthetic spiral dataset.
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate spiral dataset (3 classes)
def generate_spirals(n_points=100, n_classes=3):
    X = np.zeros((n_points * n_classes, 2))
    y = np.zeros(n_points * n_classes, dtype=int)
    for c in range(n_classes):
        ix = range(n_points * c, n_points * (c + 1))
        r = np.linspace(0.0, 1, n_points)
        t = np.linspace(c * 4, (c + 1) * 4, n_points) + np.random.randn(n_points) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = c
    return X, y

X_train, y_train = generate_spirals(100, 3)
print(f"Data: {X_train.shape}, Labels: {y_train.shape}")

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        exp_z = np.exp(self.z2 - np.max(self.z2, axis=-1, keepdims=True))
        self.a2 = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
        return self.a2

    def compute_loss(self, probs, y):
        correct = probs[np.arange(len(y)), y]
        return -np.mean(np.log(np.clip(correct, 1e-15, 1.0)))

    def backward(self, y):
        batch_size = len(y)
        y_oh = np.zeros_like(self.a2)
        y_oh[np.arange(batch_size), y] = 1.0
        dz2 = (self.a2 - y_oh) / batch_size
        self.dW2 = self.a1.T @ dz2
        self.db2 = np.sum(dz2, axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)
        self.dW1 = self.X.T @ dz1
        self.db1 = np.sum(dz1, axis=0)

    def update(self, lr):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2

# Training loop
net = TwoLayerNet(2, 100, 3)
losses = []

for epoch in range(1000):
    probs = net.forward(X_train)
    loss = net.compute_loss(probs, y_train)
    net.backward(y_train)
    net.update(lr=1.0)
    losses.append(loss)
    if epoch % 100 == 0:
        preds = np.argmax(probs, axis=1)
        acc = np.mean(preds == y_train)
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {acc:.2%}")

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(losses)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')

# Decision boundary
h = 0.01
x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = np.argmax(net.forward(grid), axis=1).reshape(xx.shape)
ax2.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
ax2.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolors='black', s=20)
ax2.set_title('Decision Boundary')

plt.tight_layout()
plt.savefig('week05_backprop/day28_spiral_training.png', dpi=150)
plt.show()
print("✅ Neural network trained on spirals from scratch!")
```

### ✅ Day 28 Checklist
- [ ] I have a COMPLETE training loop: forward → loss → backward → update
- [ ] My network learns the spiral dataset (loss decreases, accuracy increases)
- [ ] I plotted the decision boundary and it correctly separates the 3 classes
- [ ] Code is pushed to GitHub

---

## 📅 Day 29 — Friday, Sep 5: Mini-Batch Training & Data Preprocessing

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week05_backprop/day29_minibatch_preprocessing.py`

**Exercise 1:** Add mini-batch training.
```python
import numpy as np

def create_batches(X, y, batch_size, shuffle=True):
    """Yield mini-batches of (X_batch, y_batch)"""
    n = len(y)
    if shuffle:
        indices = np.random.permutation(n)
        X, y = X[indices], y[indices]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield X[start:end], y[start:end]

# Training with mini-batches
X_train, y_train = generate_spirals(100, 3)
net = TwoLayerNet(2, 100, 3)

for epoch in range(200):
    epoch_loss = 0
    n_batches = 0
    for X_batch, y_batch in create_batches(X_train, y_train, batch_size=32):
        probs = net.forward(X_batch)
        loss = net.compute_loss(probs, y_batch)
        net.backward(y_batch)
        net.update(lr=0.5)
        epoch_loss += loss
        n_batches += 1
    if epoch % 20 == 0:
        full_probs = net.forward(X_train)
        acc = np.mean(np.argmax(full_probs, axis=1) == y_train)
        print(f"Epoch {epoch:3d} | Loss: {epoch_loss/n_batches:.4f} | Acc: {acc:.2%}")
```

**Exercise 2:** Data preprocessing pipeline.
```python
def preprocess(X_train, X_test):
    """Standard preprocessing: normalize to zero mean, unit variance"""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8  # Prevent division by zero
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std  # Use TRAIN stats on test data!
    return X_train_norm, X_test_norm, mean, std

def one_hot_encode(y, n_classes):
    """Convert integer labels to one-hot vectors"""
    one_hot = np.zeros((len(y), n_classes))
    one_hot[np.arange(len(y)), y] = 1.0
    return one_hot

def train_test_split(X, y, test_ratio=0.2, seed=42):
    """Split data into train and test sets"""
    rng = np.random.default_rng(seed)
    n = len(y)
    indices = rng.permutation(n)
    split = int(n * (1 - test_ratio))
    return X[indices[:split]], X[indices[split:]], y[indices[:split]], y[indices[split:]]

# Test the pipeline
X, y = generate_spirals(200, 3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)
X_train_n, X_test_n, _, _ = preprocess(X_train, X_test)

print(f"Train: {X_train_n.shape}, Test: {X_test_n.shape}")
print(f"Train mean: {X_train_n.mean(axis=0)}")  # Should be ~0
print(f"Train std:  {X_train_n.std(axis=0)}")   # Should be ~1
print("✅ Preprocessing pipeline complete!")
```

### ✅ Day 29 Checklist
- [ ] I implemented mini-batch training with shuffling
- [ ] I built a preprocessing pipeline (normalize, one-hot, train/test split)
- [ ] I understand: ALWAYS compute mean/std from TRAINING data only
- [ ] Code is pushed to GitHub

---

## 📅 Day 30 — Saturday, Sep 6: Week 5 Mastery Test

### 🌙 Evening — Comprehensive Review

**File:** `week05_backprop/day30_mastery_test.py`

**The Test:** Without looking at any previous code, implement a complete neural network from scratch that:
1. Creates synthetic data (spirals or moons)
2. Preprocesses it (normalize, split)
3. Defines a 2-layer network (forward + backward)
4. Trains with mini-batch SGD
5. Evaluates on test set
6. Plots loss curve and decision boundary

**If you can do all of this from memory → you've mastered backpropagation.**
**If you get stuck → go back to Days 25-29 and redo them.**

### ✅ Day 30 Checklist
- [ ] I implemented everything from scratch without looking at old code
- [ ] My network achieves >90% accuracy on the test set
- [ ] I can explain every line of the backward pass
- [ ] Code is pushed to GitHub

---

> 🌙 **Sunday Sep 7 — REST DAY.**

---

# WEEK 6 — PROBABILITY & STATISTICS FOUNDATIONS
### Sep 8 (Mon) – Sep 13 (Sat)

---

## 📅 Day 31 — Monday, Sep 8: Probability Foundations & Bayes' Theorem

### 🌅 Morning (6:00 – 7:30 AM) — Watch

**What to watch:** 3B1B "Bayes theorem" + "The quick proof of Bayes' theorem"

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week06_probability/day31_probability_basics.py`

**Exercise 1:** Simulate and verify probability rules.
```python
import numpy as np

rng = np.random.default_rng(42)

# Simulate 100,000 dice rolls
n = 100_000
rolls = rng.integers(1, 7, size=n)

# P(6) should be ~1/6
p_six = np.mean(rolls == 6)
print(f"P(6) = {p_six:.4f} (expected: {1/6:.4f})")

# P(even) should be ~0.5
p_even = np.mean(rolls % 2 == 0)
print(f"P(even) = {p_even:.4f} (expected: 0.5)")

# P(>4) should be ~1/3
p_gt4 = np.mean(rolls > 4)
print(f"P(>4) = {p_gt4:.4f} (expected: {1/3:.4f})")
```

**Exercise 2:** Implement Bayes' Theorem — Medical Test Example.
```python
# Disease prevalence: 1% of population has the disease
# Test sensitivity (true positive rate): 99%
# Test specificity (true negative rate): 95%
# Question: If you test positive, what's the probability you have the disease?

p_disease = 0.01
p_positive_given_disease = 0.99    # Sensitivity
p_positive_given_healthy = 0.05    # 1 - Specificity (false positive rate)

# P(positive) = P(pos|disease)*P(disease) + P(pos|healthy)*P(healthy)
p_positive = p_positive_given_disease * p_disease + p_positive_given_healthy * (1 - p_disease)

# Bayes: P(disease|positive) = P(pos|disease) * P(disease) / P(positive)
p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive

print(f"P(disease | positive test) = {p_disease_given_positive:.4f}")
print(f"That's only {p_disease_given_positive*100:.1f}%!")
print("Even with a 99% accurate test, a positive result only means ~17% chance of disease!")
print("This is the BASE RATE FALLACY — the prior (1% prevalence) matters enormously.")

# Verify by simulation
n = 1_000_000
has_disease = rng.random(n) < p_disease
test_result = np.where(
    has_disease,
    rng.random(n) < p_positive_given_disease,
    rng.random(n) < p_positive_given_healthy
)
simulated = np.mean(has_disease[test_result])
print(f"\nSimulated P(disease|positive) = {simulated:.4f}")
print("✅ Matches Bayes' theorem!")
```

### ✅ Day 31 Checklist
- [ ] I can state Bayes' theorem from memory
- [ ] I understand the base rate fallacy
- [ ] I verified Bayes' theorem with simulation
- [ ] Code is pushed to GitHub

---

## 📅 Day 32 — Tuesday, Sep 9: Discrete Distributions

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week06_probability/day32_discrete_distributions.py`

**Exercise 1:** Implement and visualize Bernoulli, Binomial, and Poisson distributions.
```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Bernoulli: Single coin flip
p = 0.7
bernoulli = rng.random(10000) < p
axes[0].hist(bernoulli.astype(int), bins=[-0.5, 0.5, 1.5], density=True, edgecolor='black')
axes[0].set_title(f'Bernoulli(p={p})')
axes[0].set_xticks([0, 1])
print(f"Bernoulli mean: {bernoulli.mean():.3f} (expected: {p})")

# Binomial: n coin flips, count successes
n_trials, p = 20, 0.5
binomial = rng.binomial(n_trials, p, 10000)
axes[1].hist(binomial, bins=range(0, n_trials+2), density=True, edgecolor='black')
axes[1].set_title(f'Binomial(n={n_trials}, p={p})')
print(f"Binomial mean: {binomial.mean():.3f} (expected: {n_trials*p})")

# Poisson: Events per time interval
lam = 5
poisson = rng.poisson(lam, 10000)
axes[2].hist(poisson, bins=range(0, 20), density=True, edgecolor='black')
axes[2].set_title(f'Poisson(λ={lam})')
print(f"Poisson mean: {poisson.mean():.3f} (expected: {lam})")

plt.tight_layout()
plt.savefig('week06_probability/day32_distributions.png', dpi=150)
plt.show()
```

### ✅ Day 32 Checklist
- [ ] I can explain Bernoulli, Binomial, and Poisson in one sentence each
- [ ] I know: Binomial mean = np, Poisson mean = λ
- [ ] Code is pushed to GitHub

---

## 📅 Day 33 — Wednesday, Sep 10: Gaussian Distribution & Central Limit Theorem

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week06_probability/day33_gaussian_clt.py`

**Exercise 1:** Demonstrate the Central Limit Theorem.
```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Central Limit Theorem: Sample Means → Normal Distribution', fontsize=14)

# Start with a UNIFORM distribution (definitely NOT normal!)
population = rng.uniform(0, 10, 100_000)
axes[0, 0].hist(population, bins=50, density=True, alpha=0.7)
axes[0, 0].set_title('Population (Uniform)')

sample_sizes = [1, 2, 5, 10, 30]
for i, n in enumerate(sample_sizes):
    # Take 10000 samples of size n, compute their means
    sample_means = np.array([rng.choice(population, n).mean() for _ in range(10000)])
    row, col = (i + 1) // 3, (i + 1) % 3
    axes[row, col].hist(sample_means, bins=50, density=True, alpha=0.7, color='coral')
    axes[row, col].set_title(f'Sample means (n={n})')

plt.tight_layout()
plt.savefig('week06_probability/day33_clt.png', dpi=150)
plt.show()
print("CLT: Even from a uniform distribution, sample means become normal as n increases!")
```

**Exercise 2:** Implement the Gaussian PDF from scratch.
```python
def gaussian_pdf(x, mu, sigma):
    """The famous bell curve formula"""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

x = np.linspace(-4, 4, 1000)
plt.figure(figsize=(10, 5))
for sigma in [0.5, 1.0, 2.0]:
    plt.plot(x, gaussian_pdf(x, 0, sigma), label=f'σ={sigma}', linewidth=2)
plt.legend()
plt.title('Gaussian PDF with different σ')
plt.grid(True, alpha=0.3)
plt.savefig('week06_probability/day33_gaussian.png', dpi=150)
plt.show()
```

### ✅ Day 33 Checklist
- [ ] I can write the Gaussian PDF formula from memory
- [ ] I demonstrated CLT: sample means → normal, regardless of original distribution
- [ ] I know the 68-95-99.7 rule
- [ ] Code is pushed to GitHub

---

## 📅 Day 34 — Thursday, Sep 11: Entropy & Cross-Entropy

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week06_probability/day34_entropy.py`

**Exercise 1:** Implement entropy, cross-entropy, and KL divergence.
```python
import numpy as np

def entropy(p):
    """H(p) = -Σ p(x) * log(p(x))"""
    p = np.clip(p, 1e-15, 1.0)
    return -np.sum(p * np.log(p))

def cross_entropy(p, q):
    """H(p, q) = -Σ p(x) * log(q(x))"""
    q = np.clip(q, 1e-15, 1.0)
    return -np.sum(p * np.log(q))

def kl_divergence(p, q):
    """KL(p || q) = Σ p(x) * log(p(x) / q(x)) = H(p,q) - H(p)"""
    return cross_entropy(p, q) - entropy(p)

# Test: Fair coin vs biased coin
fair = np.array([0.5, 0.5])
biased = np.array([0.9, 0.1])
very_biased = np.array([0.99, 0.01])

print(f"Entropy of fair coin:        {entropy(fair):.4f}")        # Maximum: log(2) = 0.693
print(f"Entropy of biased (90/10):   {entropy(biased):.4f}")      # Lower
print(f"Entropy of very biased:      {entropy(very_biased):.4f}") # Very low

print(f"\nKL(fair || biased):   {kl_divergence(fair, biased):.4f}")
print(f"KL(biased || fair):   {kl_divergence(biased, fair):.4f}")
print("Note: KL is NOT symmetric! KL(p||q) ≠ KL(q||p)")

# KEY INSIGHT for ML:
# When we minimize cross-entropy H(p_true, p_model), we're minimizing
# KL(p_true || p_model) because H(p_true) is constant!
# That's why cross-entropy loss works as a training objective.
print("\n✅ Cross-entropy loss = minimizing KL divergence from true distribution!")
```

### ✅ Day 34 Checklist
- [ ] I can implement entropy, cross-entropy, and KL divergence from scratch
- [ ] I understand: higher entropy = more uncertainty/randomness
- [ ] I understand: KL divergence is NOT symmetric
- [ ] I understand: minimizing CE loss = minimizing KL divergence
- [ ] Code is pushed to GitHub

---

## 📅 Day 35 — Friday, Sep 12: Maximum Likelihood Estimation

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week06_probability/day35_mle.py`

**Exercise 1:** MLE for Gaussian parameters.
```python
import numpy as np

# Generate data from known Gaussian
true_mu, true_sigma = 5.0, 2.0
rng = np.random.default_rng(42)
data = rng.normal(true_mu, true_sigma, 1000)

# MLE estimates (for Gaussian, MLE = sample mean and sample std)
mle_mu = np.mean(data)
mle_sigma = np.std(data)  # Note: MLE uses N, not N-1

print(f"True: μ={true_mu}, σ={true_sigma}")
print(f"MLE:  μ={mle_mu:.4f}, σ={mle_sigma:.4f}")
print("MLE converges to true parameters with enough data!")

# Show convergence with different sample sizes
for n in [10, 50, 100, 500, 1000, 10000]:
    sample = rng.normal(true_mu, true_sigma, n)
    print(f"  n={n:5d}: μ̂={np.mean(sample):.3f}, σ̂={np.std(sample):.3f}")
```

**Exercise 2:** MLE for logistic regression (this IS what your network does!).
```python
# Key insight: Training a neural network with cross-entropy loss
# IS maximum likelihood estimation.
#
# P(y|x; θ) = softmax(f(x; θ))
# log P(data|θ) = Σ log P(y_i | x_i; θ) = Σ log softmax(f(x_i; θ))[y_i]
# Maximizing log-likelihood = Minimizing cross-entropy loss
#
# YOUR NEURAL NETWORK IS DOING MLE!

print("\n🎯 KEY INSIGHT:")
print("  Cross-entropy loss = negative log-likelihood")
print("  Training neural networks = Maximum Likelihood Estimation")
print("  They're the SAME THING, just different names!")
```

### ✅ Day 35 Checklist
- [ ] I can compute MLE estimates for Gaussian parameters
- [ ] I understand: training with CE loss = MLE
- [ ] Code is pushed to GitHub

---

## 📅 Day 36 — Saturday, Sep 13: Week 6 Review & Probability Mastery Test

**File:** `week06_probability/day36_review.py`

Without looking at notes, implement:
1. Bayes' theorem for a medical test scenario
2. Simulate CLT from an exponential distribution
3. Compute entropy and KL divergence for two distributions
4. Explain how cross-entropy loss relates to MLE

### ✅ Day 36 Checklist
- [ ] I can do all 4 tasks from memory
- [ ] I understand the connection: probability → information theory → loss functions → ML training

---

> 🌙 **Sunday Sep 14 — REST DAY.**

---

# WEEK 7 — CLASSICAL ML ALGORITHMS FROM SCRATCH
### Sep 15 (Mon) – Sep 20 (Sat)

---

## 📅 Day 37 — Monday, Sep 15: Linear Regression from Scratch

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week07_classical_ml/day37_linear_regression.py`

**Exercise 1:** Implement linear regression with gradient descent AND the normal equation.
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data: y = 3x + 7 + noise
rng = np.random.default_rng(42)
X = rng.uniform(0, 10, (100, 1))
y = 3 * X.squeeze() + 7 + rng.normal(0, 2, 100)

# Method 1: Normal Equation (closed-form solution)
# θ = (X^T X)^(-1) X^T y
X_b = np.c_[np.ones(100), X]  # Add bias column
theta_normal = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print(f"Normal Equation: w={theta_normal[1]:.4f}, b={theta_normal[0]:.4f}")

# Method 2: Gradient Descent
w, b = 0.0, 0.0
lr = 0.01
losses = []

for epoch in range(1000):
    y_pred = w * X.squeeze() + b
    loss = np.mean((y_pred - y) ** 2)
    losses.append(loss)

    # Gradients
    dw = (2 / len(y)) * np.sum((y_pred - y) * X.squeeze())
    db = (2 / len(y)) * np.sum(y_pred - y)

    w -= lr * dw
    b -= lr * db

print(f"Gradient Descent: w={w:.4f}, b={b:.4f}")
print(f"True values:      w=3.0000, b=7.0000")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(X, y, alpha=0.5, s=10)
ax1.plot(X, w * X + b, 'r-', linewidth=2, label=f'y={w:.2f}x+{b:.2f}')
ax1.legend()
ax1.set_title('Linear Regression Fit')
ax2.plot(losses)
ax2.set_title('Training Loss')
ax2.set_xlabel('Epoch')
plt.tight_layout()
plt.savefig('week07_classical_ml/day37_linear_regression.png', dpi=150)
plt.show()
```

### ✅ Day 37 Checklist
- [ ] I solved linear regression with both normal equation and gradient descent
- [ ] Both give the same answer (approximately)
- [ ] I understand: normal equation = O(n³), GD = O(n·iterations)
- [ ] Code is pushed to GitHub

---

## 📅 Day 38 — Tuesday, Sep 16: Logistic Regression from Scratch

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week07_classical_ml/day38_logistic_regression.py`

**Exercise 1:** Implement binary logistic regression.
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate 2-class data
rng = np.random.default_rng(42)
X_pos = rng.normal([2, 2], 1.0, (100, 2))
X_neg = rng.normal([-2, -2], 1.0, (100, 2))
X = np.vstack([X_pos, X_neg])
y = np.array([1]*100 + [0]*100)

# Shuffle
idx = rng.permutation(200)
X, y = X[idx], y[idx]

# Logistic Regression
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

w = np.zeros(2)
b = 0.0
lr = 0.1
losses = []

for epoch in range(500):
    z = X @ w + b
    y_pred = sigmoid(z)

    # Binary cross-entropy loss
    loss = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
    losses.append(loss)

    # Gradients
    error = y_pred - y
    dw = (1 / len(y)) * X.T @ error
    db = (1 / len(y)) * np.sum(error)

    w -= lr * dw
    b -= lr * db

    if epoch % 50 == 0:
        acc = np.mean((y_pred > 0.5) == y)
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Acc: {acc:.2%}")

print(f"\nFinal weights: {w}, bias: {b:.4f}")
print(f"Final accuracy: {np.mean((sigmoid(X @ w + b) > 0.5) == y):.2%}")
```

### ✅ Day 38 Checklist
- [ ] I implemented logistic regression from scratch
- [ ] I understand: it's a single neuron with sigmoid + BCE loss
- [ ] My model achieves >95% accuracy on the 2-class problem
- [ ] Code is pushed to GitHub

---

## 📅 Day 39 — Wednesday, Sep 17: Naive Bayes from Scratch

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week07_classical_ml/day39_naive_bayes.py`

**Exercise 1:** Implement Gaussian Naive Bayes.
```python
import numpy as np

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.means = {}    # Mean of each feature for each class
        self.vars = {}     # Variance of each feature for each class
        self.priors = {}   # P(class)

        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = X_c.mean(axis=0)
            self.vars[c] = X_c.var(axis=0) + 1e-9  # Add smoothing
            self.priors[c] = len(X_c) / len(y)

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                # Log probability (avoids underflow from multiplying many small numbers)
                log_prior = np.log(self.priors[c])
                log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.vars[c])) \
                                 -0.5 * np.sum((x - self.means[c])**2 / self.vars[c])
                posteriors.append(log_prior + log_likelihood)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)

# Test on Iris-like data
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

# Manual train/test split
rng = np.random.default_rng(42)
idx = rng.permutation(len(y))
X_train, X_test = X[idx[:120]], X[idx[120:]]
y_train, y_test = y[idx[:120]], y[idx[120:]]

nb = GaussianNaiveBayes()
nb.fit(X_train, y_train)
preds = nb.predict(X_test)
acc = np.mean(preds == y_test)
print(f"Naive Bayes accuracy: {acc:.2%}")
```

### ✅ Day 39 Checklist
- [ ] I implemented Gaussian Naive Bayes from scratch
- [ ] I understand: "naive" = features are assumed independent
- [ ] I used log probabilities to avoid numerical underflow
- [ ] Code is pushed to GitHub

---

## 📅 Day 40 — Thursday, Sep 18: Sampling — Temperature, Top-k, Top-p

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week07_classical_ml/day40_sampling.py`

**Exercise 1:** Implement all three sampling strategies (used in EVERY LLM).
```python
import numpy as np

def sample_with_temperature(logits, temperature=1.0):
    """Higher temp = more random, Lower temp = more deterministic"""
    scaled = logits / temperature
    probs = np.exp(scaled - np.max(scaled))
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs), probs

def sample_top_k(logits, k=5):
    """Only sample from the top-k highest probability tokens"""
    top_k_idx = np.argsort(logits)[-k:]
    top_k_logits = logits[top_k_idx]
    probs = np.exp(top_k_logits - np.max(top_k_logits))
    probs /= probs.sum()
    chosen = np.random.choice(len(probs), p=probs)
    return top_k_idx[chosen], probs

def sample_top_p(logits, p=0.9):
    """Sample from smallest set of tokens whose cumulative probability >= p"""
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()
    sorted_idx = np.argsort(probs)[::-1]
    cumsum = np.cumsum(probs[sorted_idx])
    cutoff = np.searchsorted(cumsum, p) + 1
    top_p_idx = sorted_idx[:cutoff]
    top_p_probs = probs[top_p_idx]
    top_p_probs /= top_p_probs.sum()
    chosen = np.random.choice(len(top_p_probs), p=top_p_probs)
    return top_p_idx[chosen], top_p_probs

# Test with fake vocabulary of 10 tokens
logits = np.array([2.0, 1.5, 3.0, 0.5, 0.1, 2.5, 0.3, 1.0, 0.2, 4.0])
vocab = ["the", "cat", "dog", "is", "a", "big", "sat", "on", "mat", "ran"]

print("=== Temperature Sampling ===")
for temp in [0.1, 0.5, 1.0, 2.0]:
    tokens = [vocab[sample_with_temperature(logits, temp)[0]] for _ in range(10)]
    print(f"  T={temp}: {tokens}")

print("\n=== Top-k Sampling (k=3) ===")
for _ in range(5):
    idx, _ = sample_top_k(logits, k=3)
    print(f"  Selected: {vocab[idx]}")

print("\n=== Top-p Sampling (p=0.9) ===")
for _ in range(5):
    idx, _ = sample_top_p(logits, p=0.9)
    print(f"  Selected: {vocab[idx]}")
```

### ✅ Day 40 Checklist
- [ ] I implemented temperature, top-k, and top-p sampling
- [ ] I understand: temperature=0.1 → nearly deterministic, temperature=2.0 → very random
- [ ] I understand: these are used in EVERY LLM (GPT, Claude, etc.)
- [ ] Code is pushed to GitHub

---

## 📅 Day 41 — Friday, Sep 19: Weight Initialization & Regularization

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week07_classical_ml/day41_initialization_regularization.py`

**Exercise 1:** Compare weight initialization strategies.
```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_forward_pass(init_fn, n_layers=10, n_neurons=256, n_samples=100):
    """Pass random data through n_layers and track activations"""
    x = np.random.randn(n_samples, n_neurons)
    activation_means = []
    activation_stds = []

    for _ in range(n_layers):
        W = init_fn(n_neurons, n_neurons)
        x = np.maximum(0, x @ W)  # ReLU
        activation_means.append(x.mean())
        activation_stds.append(x.std())

    return activation_means, activation_stds

# Random init (bad)
random_init = lambda fan_in, fan_out: np.random.randn(fan_in, fan_out) * 0.01
# Xavier init (good for sigmoid/tanh)
xavier_init = lambda fan_in, fan_out: np.random.randn(fan_in, fan_out) * np.sqrt(1.0 / fan_in)
# He init (good for ReLU)
he_init = lambda fan_in, fan_out: np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, (name, init_fn) in enumerate([("Random (0.01)", random_init), ("Xavier", xavier_init), ("He", he_init)]):
    means, stds = simulate_forward_pass(init_fn)
    axes[i].bar(range(10), stds, color=['#4A90D9' if s > 0.1 else '#FF6B6B' for s in stds])
    axes[i].set_title(f"{name}")
    axes[i].set_xlabel("Layer")
    axes[i].set_ylabel("Activation Std")
    axes[i].set_ylim(0, max(stds) * 1.2 + 0.1)

plt.suptitle("Activation statistics through 10 layers (ReLU)")
plt.tight_layout()
plt.savefig('week07_classical_ml/day41_initialization.png', dpi=150)
plt.show()
print("He init keeps activations alive through many layers!")
```

**Exercise 2:** Implement L2 regularization (weight decay).
```python
def train_with_regularization(X, y, hidden=50, lr=0.1, reg_lambda=0.01, epochs=500):
    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    W1 = np.random.randn(n_features, hidden) * np.sqrt(2.0 / n_features)
    b1 = np.zeros(hidden)
    W2 = np.random.randn(hidden, n_classes) * np.sqrt(2.0 / hidden)
    b2 = np.zeros(n_classes)

    for epoch in range(epochs):
        # Forward
        z1 = X @ W1 + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2 + b2
        exp_z = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        # Loss with L2 regularization
        n = len(y)
        correct = probs[np.arange(n), y]
        data_loss = -np.mean(np.log(np.clip(correct, 1e-15, 1.0)))
        reg_loss = 0.5 * reg_lambda * (np.sum(W1**2) + np.sum(W2**2))
        total_loss = data_loss + reg_loss

        # Backward (with L2 gradient added)
        y_oh = np.zeros_like(probs)
        y_oh[np.arange(n), y] = 1.0
        dz2 = (probs - y_oh) / n
        dW2 = a1.T @ dz2 + reg_lambda * W2  # ← L2 gradient
        db2 = np.sum(dz2, axis=0)
        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1 + reg_lambda * W1    # ← L2 gradient
        db1 = np.sum(dz1, axis=0)

        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2

    return total_loss

print("L2 regularization = adding λ·W to every weight gradient")
print("This penalizes large weights, preventing overfitting!")
```

### ✅ Day 41 Checklist
- [ ] I understand Xavier (for sigmoid/tanh) vs He (for ReLU) initialization
- [ ] I saw: bad init → activations die or explode through layers
- [ ] I implemented L2 regularization (weight decay)
- [ ] Code is pushed to GitHub

---

## 📅 Day 42 — Saturday, Sep 20: Dropout & Week 7 Review

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week07_classical_ml/day42_dropout.py`

**Exercise 1:** Implement dropout from scratch.
```python
import numpy as np

def dropout_forward(x, keep_prob=0.8, training=True):
    """
    During training: randomly zero out neurons with probability (1-keep_prob)
    During inference: use all neurons but scale by keep_prob
    """
    if not training:
        return x, None

    # Create mask: 1 with probability keep_prob, 0 otherwise
    mask = (np.random.rand(*x.shape) < keep_prob).astype(float)
    # Apply mask and scale (inverted dropout — so we don't need to scale at inference)
    out = x * mask / keep_prob
    return out, mask

def dropout_backward(dout, mask, keep_prob):
    return dout * mask / keep_prob

# Demo
x = np.ones((1, 10))
print(f"Input:                {x}")
dropped, mask = dropout_forward(x, keep_prob=0.5, training=True)
print(f"After dropout (train):{dropped}")
print(f"Mask:                 {mask}")
intact, _ = dropout_forward(x, keep_prob=0.5, training=False)
print(f"After dropout (eval): {intact}")
print("\nInverted dropout: scale during training so inference is unchanged!")
```

### ✅ Day 42 Checklist
- [ ] I implemented dropout with inverted scaling
- [ ] I understand: dropout during training, full network during inference
- [ ] I understand: dropout = implicit ensemble of sub-networks
- [ ] Code is pushed to GitHub

---

> 🌙 **Sunday Sep 21 — REST DAY.**

---

# WEEK 8-9 — BUILDING THE NEURAL NETWORK LIBRARY
### Sep 22 (Mon) – Oct 4 (Sat)
> *You take everything from Weeks 1-7 and build a clean, reusable neural network library. This becomes the capstone foundation.*

---

## 📅 Day 43 — Monday, Sep 22: Design the NumForge Library

### 🌙 Evening (7:00 – 9:00 PM) — Code

**File:** `week08_numforge/numforge/__init__.py` + `layers.py`

**Exercise 1:** Design the Layer abstraction.
```python
# numforge/layers.py
import numpy as np

class Layer:
    """Base class for all layers"""
    def forward(self, x):
        raise NotImplementedError
    def backward(self, grad):
        raise NotImplementedError
    def params(self):
        return []

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros(output_size)
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW = self.x.T @ grad
        self.db = np.sum(grad, axis=0)
        return grad @ self.W.T

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]

class ReLU(Layer):
    def forward(self, x):
        self.mask = (x > 0).astype(float)
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask

class Softmax(Layer):
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.out

    def backward(self, grad):
        return grad  # Combined with CE loss

class Dropout(Layer):
    def __init__(self, keep_prob=0.8):
        self.keep_prob = keep_prob
        self.training = True

    def forward(self, x):
        if self.training:
            self.mask = (np.random.rand(*x.shape) < self.keep_prob).astype(float)
            return x * self.mask / self.keep_prob
        return x

    def backward(self, grad):
        if self.training:
            return grad * self.mask / self.keep_prob
        return grad
```

### ✅ Day 43 Checklist
- [ ] I designed a clean Layer interface with forward/backward/params
- [ ] I implemented Dense, ReLU, Softmax, and Dropout layers
- [ ] Each layer stores what it needs for backward pass
- [ ] Code is pushed to GitHub

---

## 📅 Day 44 — Tuesday, Sep 23: Loss Functions & Optimizers

**File:** `week08_numforge/numforge/losses.py` + `optimizers.py`

**Exercise 1:** Implement loss functions as classes.
```python
# numforge/losses.py
import numpy as np

class CrossEntropyLoss:
    def forward(self, probs, y_true):
        self.probs = probs
        self.y_true = y_true
        batch_size = len(y_true)
        correct = probs[np.arange(batch_size), y_true]
        return -np.mean(np.log(np.clip(correct, 1e-15, 1.0)))

    def backward(self):
        batch_size = len(self.y_true)
        grad = self.probs.copy()
        grad[np.arange(batch_size), self.y_true] -= 1.0
        return grad / batch_size
```

**Exercise 2:** Implement optimizers.
```python
# numforge/optimizers.py
import numpy as np

class SGD:
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            for j, (param, grad) in enumerate(layer.params()):
                key = (i, j)
                if key not in self.velocities:
                    self.velocities[key] = np.zeros_like(param)
                self.velocities[key] = self.momentum * self.velocities[key] - self.lr * grad
                param += self.velocities[key]

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, layers):
        self.t += 1
        for i, layer in enumerate(layers):
            for j, (param, grad) in enumerate(layer.params()):
                key = (i, j)
                if key not in self.m:
                    self.m[key] = np.zeros_like(param)
                    self.v[key] = np.zeros_like(param)
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grad**2
                m_hat = self.m[key] / (1 - self.beta1**self.t)
                v_hat = self.v[key] / (1 - self.beta2**self.t)
                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
```

### ✅ Day 44 Checklist
- [ ] I implemented CrossEntropyLoss with forward and backward
- [ ] I implemented SGD with momentum and Adam optimizers
- [ ] Both optimizers update parameters in-place
- [ ] Code is pushed to GitHub

---

## 📅 Day 45 — Wednesday, Sep 24: Network Class & Training API

**File:** `week08_numforge/numforge/network.py`

```python
import numpy as np

class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train_mode(self):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True

    def eval_mode(self):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False

    def train(self, X, y, loss_fn, optimizer, epochs, batch_size=32, verbose=True):
        history = []
        n = len(y)
        for epoch in range(epochs):
            self.train_mode()
            idx = np.random.permutation(n)
            epoch_loss = 0
            n_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                X_b = X[idx[start:end]]
                y_b = y[idx[start:end]]

                probs = self.forward(X_b)
                loss = loss_fn.forward(probs, y_b)
                grad = loss_fn.backward()
                self.backward(grad)
                optimizer.step(self.layers)

                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            history.append(avg_loss)

            if verbose and epoch % (epochs // 10) == 0:
                self.eval_mode()
                preds = np.argmax(self.forward(X), axis=1)
                acc = np.mean(preds == y)
                print(f"Epoch {epoch:4d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2%}")

        return history
```

### ✅ Day 45 Checklist
- [ ] My Network class has a clean train() method
- [ ] It supports mini-batches, shuffling, and any optimizer
- [ ] train_mode/eval_mode toggle dropout correctly
- [ ] Code is pushed to GitHub

---

## 📅 Day 46-48 — Thu Sep 25 – Sat Sep 27: Test & Debug the Library

**Exercise:** Train NumForge on the spiral dataset from Day 28. Then on a harder dataset (sklearn moons/circles). Fix any bugs.

```python
from numforge.layers import Dense, ReLU, Softmax, Dropout
from numforge.losses import CrossEntropyLoss
from numforge.optimizers import Adam
from numforge.network import Network

net = Network([
    Dense(2, 128), ReLU(), Dropout(0.8),
    Dense(128, 64), ReLU(), Dropout(0.8),
    Dense(64, 3), Softmax()
])

loss_fn = CrossEntropyLoss()
optimizer = Adam(lr=0.01)
history = net.train(X_train, y_train, loss_fn, optimizer, epochs=500, batch_size=32)
```

### ✅ Days 46-48 Checklist
- [ ] NumForge works end-to-end on spiral dataset
- [ ] I tested with different architectures (wider, deeper)
- [ ] I tested with different optimizers (SGD vs Adam)
- [ ] All edge cases handled (single sample, large batch, etc.)
- [ ] Code is pushed to GitHub

---

> 🌙 **Sunday Sep 28 — REST DAY.**

---

# WEEK 10 — THE CAPSTONE: MNIST FROM SCRATCH
### Sep 29 (Mon) – Oct 4 (Sat)
> *This is IT. You train a neural network on real data — the MNIST handwritten digits dataset — using ONLY your NumForge library. Target: >95% accuracy.*

---

## 📅 Day 49 — Monday, Sep 29: Load & Explore MNIST

**File:** `week10_capstone/day49_load_mnist.py`

```python
import numpy as np
import matplotlib.pyplot as plt

# Download MNIST (using sklearn for convenience — this is just data loading)
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_all, y_all = mnist.data.astype(np.float64), mnist.target.astype(int)

# Normalize to [0, 1]
X_all = X_all / 255.0

# Split
X_train, X_test = X_all[:60000], X_all[60000:]
y_train, y_test = y_all[:60000], y_all[60000:]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Labels: {np.unique(y_train)}")
print(f"Pixel range: [{X_train.min()}, {X_train.max()}]")

# Visualize 25 samples
fig, axes = plt.subplots(5, 5, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis('off')
plt.tight_layout()
plt.savefig('week10_capstone/mnist_samples.png', dpi=150)
plt.show()
```

### ✅ Day 49 Checklist
- [ ] MNIST loaded and normalized to [0, 1]
- [ ] I visualized samples and verified labels match images
- [ ] Train: 60K samples, Test: 10K samples, 784 features, 10 classes
- [ ] Code is pushed to GitHub

---

## 📅 Day 50-51 — Tue-Wed, Sep 30 – Oct 1: Build & Train the Network

**File:** `week10_capstone/day50_train_mnist.py`

```python
import numpy as np
import sys
sys.path.insert(0, '../week08_numforge')
from numforge.layers import Dense, ReLU, Softmax, Dropout
from numforge.losses import CrossEntropyLoss
from numforge.optimizers import Adam
from numforge.network import Network

# Load data (from day49)
# ... (load MNIST, normalize, split)

# Architecture: 784 → 256 → 128 → 10
net = Network([
    Dense(784, 256), ReLU(), Dropout(0.8),
    Dense(256, 128), ReLU(), Dropout(0.8),
    Dense(128, 10), Softmax()
])

loss_fn = CrossEntropyLoss()
optimizer = Adam(lr=0.001)

# Train
history = net.train(X_train, y_train, loss_fn, optimizer,
                    epochs=20, batch_size=128, verbose=True)

# Evaluate
net.eval_mode()
test_probs = net.forward(X_test)
test_preds = np.argmax(test_probs, axis=1)
test_acc = np.mean(test_preds == y_test)
print(f"\n🎯 Test Accuracy: {test_acc:.2%}")
```

### ✅ Days 50-51 Checklist
- [ ] Network trains on MNIST without crashing
- [ ] Loss decreases steadily over epochs
- [ ] Training accuracy improves
- [ ] Code is pushed to GitHub

---

## 📅 Day 52-53 — Thu-Fri, Oct 2-3: Debug, Tune, Achieve >95%

**Hyperparameter tuning checklist:**
```
□ Learning rate: Try 0.01, 0.001, 0.0005, 0.0001
□ Hidden sizes: Try 128, 256, 512
□ Batch size: Try 32, 64, 128, 256
□ Dropout: Try 0.7, 0.8, 0.9, 1.0 (no dropout)
□ Epochs: Train longer if loss is still decreasing
□ Add a third hidden layer?
```

**Common bugs to check:**
```
□ Softmax numerical stability (subtract max)
□ Gradient shapes at every layer (add print statements)
□ Learning rate too high (loss explodes) or too low (loss barely moves)
□ Dropout disabled during evaluation
□ Data not shuffled between epochs
```

### ✅ Days 52-53 Checklist
- [ ] Test accuracy >95% achieved!
- [ ] I documented which hyperparameters worked best
- [ ] I can explain WHY certain choices helped
- [ ] Code is pushed to GitHub

---

## 📅 Day 54 — Saturday, Oct 4: Visualize Results & Document

**File:** `week10_capstone/day54_visualize.py`

```python
import numpy as np
import matplotlib.pyplot as plt

# Confusion matrix
from collections import Counter

def plot_confusion_matrix(y_true, y_pred, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('week10_capstone/confusion_matrix.png', dpi=150)
    plt.show()

# Show misclassified examples
wrong = np.where(test_preds != y_test)[0]
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    if i < len(wrong):
        idx = wrong[i]
        ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f"True:{y_test[idx]} Pred:{test_preds[idx]}", color='red')
    ax.axis('off')
plt.suptitle('Misclassified Examples')
plt.tight_layout()
plt.savefig('week10_capstone/misclassified.png', dpi=150)
plt.show()
```

### ✅ Day 54 Checklist
- [ ] Confusion matrix plotted — which digits are confused?
- [ ] Misclassified examples visualized — are they genuinely hard?
- [ ] Training loss curve plotted
- [ ] All results documented in README
- [ ] Code is pushed to GitHub

---

> 🌙 **Sunday Oct 5 — REST DAY. You just built a neural network from scratch that reads handwriting. Celebrate.**

---

# WEEK 11-13 — REVIEW, EINSUM & BUFFER
### Oct 6 (Mon) – Oct 31 (Fri)

---

## 📅 Day 55-56 — Mon-Tue, Oct 6-7: np.einsum Mastery

**File:** `week11_review/day55_einsum.py`

```python
import numpy as np

# einsum = "Einstein summation" — one function to rule them all

A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
v = np.random.randn(4)

# 1. Matrix multiply: C_ij = Σ_k A_ik * B_kj
C = np.einsum('ik,kj->ij', A, B)
assert np.allclose(C, A @ B)

# 2. Dot product: s = Σ_i v_i * v_i
s = np.einsum('i,i->', v, v)
assert np.allclose(s, np.dot(v, v))

# 3. Outer product: M_ij = a_i * b_j
a, b = np.array([1, 2, 3]), np.array([4, 5])
M = np.einsum('i,j->ij', a, b)
assert np.allclose(M, np.outer(a, b))

# 4. Batch matrix multiply: C_bij = Σ_k A_bik * B_bkj
A_batch = np.random.randn(8, 3, 4)
B_batch = np.random.randn(8, 4, 5)
C_batch = np.einsum('bik,bkj->bij', A_batch, B_batch)

# 5. Trace: tr(A) = Σ_i A_ii
sq = np.random.randn(4, 4)
tr = np.einsum('ii->', sq)
assert np.allclose(tr, np.trace(sq))

# 6. Attention: attn_bhij = Σ_k Q_bhik * K_bhjk (scaled dot product)
batch, heads, seq_len, d_k = 2, 4, 10, 64
Q = np.random.randn(batch, heads, seq_len, d_k)
K = np.random.randn(batch, heads, seq_len, d_k)
attn = np.einsum('bhik,bhjk->bhij', Q, K) / np.sqrt(d_k)
print(f"Attention shape: {attn.shape}")  # (2, 4, 10, 10)

print("✅ einsum handles ALL tensor operations with one syntax!")
```

### ✅ Days 55-56 Checklist
- [ ] I can write einsum for: matmul, dot, outer, batch matmul, trace, transpose
- [ ] I can write einsum for the attention computation
- [ ] I understand: einsum subscripts describe which indices to sum over

---

## 📅 Day 57-58 — Wed-Thu, Oct 8-9: Save/Load Models & Convolution Preview

**Exercise 1:** Save and load your trained MNIST model.
```python
def save_model(network, filepath):
    params = {}
    for i, layer in enumerate(network.layers):
        if hasattr(layer, 'W'):
            params[f'layer_{i}_W'] = layer.W
            params[f'layer_{i}_b'] = layer.b
    np.savez(filepath, **params)
    print(f"Model saved to {filepath}")

def load_model(network, filepath):
    data = np.load(filepath)
    for i, layer in enumerate(network.layers):
        if hasattr(layer, 'W'):
            layer.W = data[f'layer_{i}_W']
            layer.b = data[f'layer_{i}_b']
    print(f"Model loaded from {filepath}")
```

**Exercise 2:** Implement 2D convolution from scratch (preview for Phase 1.3).
```python
def conv2d(image, kernel):
    """Simple 2D convolution (no padding, stride=1)"""
    ih, iw = image.shape
    kh, kw = kernel.shape
    oh, ow = ih - kh + 1, iw - kw + 1
    output = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            output[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)
    return output

# Edge detection kernel
edge_kernel = np.array([[-1, -1, -1],
                         [-1,  8, -1],
                         [-1, -1, -1]])

sample_image = X_train[0].reshape(28, 28)
edges = conv2d(sample_image, edge_kernel)
print(f"Input: {sample_image.shape} → Output: {edges.shape}")
```

### ✅ Days 57-58 Checklist
- [ ] I can save and load NumPy model weights
- [ ] I understand how 2D convolution works (sliding window)
- [ ] Code is pushed to GitHub

---

## 📅 Day 59-66 — Oct 10-18: Comprehensive Review Sprint

**For each day, revisit one topic and redo the hardest exercise WITHOUT looking at old code:**

| Day | Topic | Key Exercise |
|-----|-------|-------------|
| 59 | NumPy indexing & broadcasting | Predict 10 broadcasting shapes on paper |
| 60 | Linear Algebra | Implement SVD and PCA from scratch |
| 61 | Calculus & Gradients | Implement Adam optimizer from memory |
| 62 | Backpropagation | Build 2-layer net from scratch (no NumForge) |
| 63 | Probability | Implement Bayes, entropy, KL divergence |
| 64 | Classical ML | Implement logistic regression from scratch |
| 65 | Sampling & Init | Implement top-p sampling + He init |
| 66 | Full pipeline | Load MNIST → preprocess → train → evaluate |

**Rule:** If you can't do it from memory, that topic needs more work before Phase 1.2.

### ✅ Days 59-66 Checklist
- [ ] Each day's exercise completed from memory
- [ ] Identified weak areas and re-studied them
- [ ] Confident in ALL topics

---

## 📅 Day 67-72 — Oct 20-25: Polish, Document & Blog

**Day 67-69:** Clean up your GitHub repo:
- Write comprehensive README with project description, results, and architecture diagram
- Add docstrings to all NumForge classes
- Create a `examples/` folder with usage demos

**Day 70-72:** Write your first blog post:
- "How I Built a Neural Network From Scratch in NumPy — And Got 95%+ on MNIST"
- Include: motivation, architecture diagram, key code snippets, results, lessons learned
- Publish on your blog/LinkedIn

### ✅ Days 67-72 Checklist
- [ ] GitHub repo is clean with great README
- [ ] NumForge library has docstrings
- [ ] Blog post written and published
- [ ] Code is pushed to GitHub

---

## 📅 Day 73-78 — Oct 27-31: Buffer & Preparation for Phase 1.2

**Use these days for:**
- Catching up on any days you fell behind
- Deepening understanding of weak areas
- Reading ahead about scikit-learn (Phase 1.2 topic)
- Relaxing and celebrating completing Phase 1.1!

**If you're fully caught up:** Implement one bonus feature in NumForge:
- Batch Normalization layer
- Learning rate scheduling (cosine annealing)
- Weight saving in ONNX-like format

### ✅ Days 73-78 Checklist
- [ ] All previous days completed (no gaps!)
- [ ] MNIST accuracy >95% confirmed
- [ ] GitHub contribution graph shows daily commits
- [ ] Ready for Phase 1.2!


---

# 📊 PHASE 1.1 COMPLETION CRITERIA

When you finish all 78 days, you must be able to answer YES to ALL of these:

### NumPy Mastery
- [ ] Create, reshape, index, slice any array instantly without looking anything up
- [ ] Predict broadcasting shapes on paper for ANY pair of arrays
- [ ] Use `axis` parameter correctly in ALL aggregations (sum, mean, max, argmax)
- [ ] Implement any linear algebra operation (solve, inverse, eigenvalues, SVD, PCA)
- [ ] Use `keepdims=True` correctly every time
- [ ] Handle views vs copies correctly
- [ ] Use `np.einsum` for multi-dimensional tensor operations

### Math Mastery
- [ ] Solve systems of linear equations (elimination + LU + NumPy)
- [ ] Compute and interpret eigenvalues/eigenvectors
- [ ] Perform SVD and use it for compression and PCA
- [ ] Compute numerical gradients and verify analytical ones
- [ ] Implement gradient descent with momentum and Adam from scratch
- [ ] Implement forward and backward passes for a 2-layer neural network
- [ ] Understand and implement cross-entropy, KL divergence, entropy

### The Capstone
- [ ] **Full neural network in pure NumPy with >95% accuracy on MNIST**
- [ ] Can read a paper's math notation and translate it to NumPy code
- [ ] GitHub repo with ALL implementations and a daily commit streak

> [!IMPORTANT]
> **If you cannot get >95% on MNIST with pure NumPy, do NOT move to Phase 1.2.** Go back, find the bug, fix it. This is the gatekeeper. Everything in deep learning is just a bigger, more complex version of what you built here.

---

> **What comes next:** Phase 1.2 (Classical ML — November-December 2026) will be expanded to this same level of detail when you're 2-3 weeks from completing Phase 1.1. There's no point detailing Phase 1.2 now — it would just be more planning without execution. **Focus on Phase 1.1. Execute. Ship.**
