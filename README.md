# 🔢 Linear Algebra & NumPy — From Scratch

> Phase 1.1 of my 3-year AI/ML roadmap. Building deep mathematical foundations with NumPy.

---

## 📂 Structure

```
Day01/  — NumPy Core: Creation, Indexing, Memory, Benchmarking
Day02/  — (coming soon) Slicing, Broadcasting, Vectorization
Day03/  — (coming soon) Linear Algebra Operations
...
```

## 🧪 Day 01 — NumPy Foundations (11 Exercises)

| # | Exercise | Concept |
|---|----------|---------|
| 01 | `numpy_day01_01.py` | `np.linspace` — 50 values from 0 to 2π |
| 02 | `numpy_day01_02.py` | Identity matrix — shape, dtype, nbytes |
| 03 | `numpy_day01_03.py` | `np.full` — float32 vs float64 memory comparison |
| 04 | `numpy_day01_04.py` | 3D array — `np.zeros`, ndim, size |
| 05 | `numpy_day01_05.py` | `np.arange` + `reshape` + `np.diag` |
| 06 | `numpy_day01_06.py` | dtype conversion — int32 → float64 |
| 07 | `numpy_day01_07(bonus).py` | Random number generation (modern Generator API) |
| 08 | `numpy_day01_08.py` | Python int (28 bytes) vs NumPy int (8 bytes) |
| 09 | `numpy_day01_09.py` | List (3.6 MB) vs Array (800 KB) for 100K elements |
| 10 | `numpy_day01_10.py` | Timing: list sum vs array sum |
| 11 | `numpy_day01_11.py` | **The Loop Trap**: for-loop vs `.sum()` — **287x speedup** |

## 🔑 Key Takeaway from Day 01

```
Python for-loop over array: 0.434 seconds
NumPy .sum():               0.002 seconds
Speedup:                    287x 🚀

→ NEVER loop over NumPy arrays. Use vectorized operations.
```

## 🗺️ Roadmap

This repo will grow to cover:
- **Linear Algebra**: Gaussian elimination, LU/QR decomposition, eigenvalues, SVD, PCA
- **All from scratch**: Every algorithm implemented in pure NumPy before using library functions
- **Visualizations**: 3D plots of convergence, transformations, and decompositions

## 🛠️ Tech

- Python 3.x
- NumPy

## 📄 License

MIT
